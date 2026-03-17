import sys
import os
from flask import Flask, render_template_string, request, redirect, url_for, flash
import psycopg2
import threading

# Configuration du path pour trouver le module orchestration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# On importe le nom de la nouvelle fonction complète
from orchestration.orchestration import traiter_pipeline_complet

app = Flask(__name__)
app.secret_key = "supersecretkey"

DB_CONFIG = {
    "host": "127.0.0.1", "database": "pfe_news",
    "user": "postgres", "password": "postgres", "port": "5432"
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>PFE News</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }
        .story-card { border: 1px solid #ccc; padding: 15px; margin: 20px 0; border-radius: 12px; background-color: #f9f9f9; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); }
        .article-item { margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 10px; }
        .article-item:last-child { border-bottom: none; }
        button { cursor: pointer; padding: 8px 15px; border-radius: 5px; border: 1px solid #007bff; background: #007bff; color: white; }
        input[type="text"] { padding: 8px; width: 250px; border-radius: 5px; border: 1px solid #ccc; }
    </style>
</head>
<body>
    <h1>🌍 Mes Stories d'Actualités</h1>
    
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        {% for message in messages %}<p style="color: #0056b3; background: #e7f3ff; padding: 10px; border-radius: 5px;">{{ message }}</p>{% endfor %}
      {% endif %}
    {% endwith %}

    <form action="/show-stories" method="post">
        <strong>Choisis tes intérêts :</strong><br><br>
        {% for interest in all_interests %}
            <label style="display: block; margin-bottom: 5px;">
                <input type="checkbox" name="interests" value="{{ interest }}" 
                       {% if interest in selected_interests %}checked{% endif %}> {{ interest }}
            </label>
        {% endfor %}
        <br>
        <button type="submit">Afficher les Stories</button>
    </form>

    <hr style="margin: 30px 0;">
    
    <h3>Ajouter un nouveau sujet</h3>
    <form action="/add-interest" method="post">
        <input type="text" name="new_interest" placeholder="Ex: SpaceX, Finance, IA..." required>
        <button type="submit" style="background: #28a745; border-color: #28a745;">Lancer le Pipeline</button>
    </form>

    {% if stories %}
        <hr>
        {% for interest, articles in stories.items() %}
            <div class="story-card">
                <h2 style="color: #333; border-bottom: 2px solid #007bff; display: inline-block;">{{ interest|capitalize }}</h2>
                {% if articles %}
                    {% for art in articles %}
                        <div class="article-item">
                            <strong style="font-size: 1.1em;">{{ art[0] }}</strong> <br>
                            <small style="color: #666;">Publié le : {{ art[1] }}</small>
                            <p>{{ art[2] }}</p>
                        </div>
                    {% endfor %}
                {% else %}
                    <p>Aucun résumé disponible pour le moment.</p>
                {% endif %}
            </div>
        {% endfor %}
    {% endif %}
</body>
</html>
"""

def get_db_cursor():
    conn = psycopg2.connect(**DB_CONFIG)
    return conn, conn.cursor()

@app.route('/', methods=['GET'])
def index():
    conn, cur = get_db_cursor()
    # On récupère les intérêts qui ont au moins un résumé réussi
    cur.execute("SELECT DISTINCT interest FROM article_summaries WHERE notes IS NULL OR notes = '';")
    interests = [row[0] for row in cur.fetchall()]
    cur.close(); conn.close()
    return render_template_string(HTML_TEMPLATE, all_interests=interests, selected_interests=[], stories=None)

@app.route('/add-interest', methods=['POST'])
def add_interest():
    topic = request.form.get('new_interest').strip()
    if not topic:
        return redirect(url_for('index'))

    conn, cur = get_db_cursor()
    # Vérification si l'intérêt existe déjà
    cur.execute("SELECT COUNT(*) FROM article_summaries WHERE interest = %s;", (topic.lower(),))
    exists = cur.fetchone()[0] > 0
    cur.close(); conn.close()

    if exists:
        flash(f"L'intérêt '{topic}' est déjà en base de données.")
    else:
        flash(f"🚀 Pipeline lancé pour '{topic}'. Les résultats apparaîtront d'ici quelques minutes.")
        # Lancement du thread avec la nouvelle fonction d'orchestration
        threading.Thread(target=traiter_pipeline_complet, args=(topic,)).start()
    
    return redirect(url_for('index'))

@app.route('/show-stories', methods=['POST'])
def show_stories():
    selected = request.form.getlist('interests')
    conn, cur = get_db_cursor()
    
    cur.execute("SELECT DISTINCT interest FROM article_summaries WHERE notes IS NULL OR notes = '';")
    all_ints = [row[0] for row in cur.fetchall()]
    
    stories = {}
    for interest in selected:
        # Modification : On ajoute "AND (notes IS NULL OR notes = '')" pour filtrer les [FAIL]
        cur.execute("""SELECT title, published_date, summary_fr FROM article_summaries 
                        WHERE interest = %s 
                        AND (notes IS NULL OR notes = '')
                        AND writing_run_id = 
                        (SELECT MAX(writing_run_id) FROM article_summaries WHERE interest = %s)
                        ORDER BY published_date DESC LIMIT 3;""", (interest, interest))
        stories[interest] = cur.fetchall()
        
    cur.close(); conn.close()
    return render_template_string(HTML_TEMPLATE, all_interests=all_ints, selected_interests=selected, stories=stories)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)