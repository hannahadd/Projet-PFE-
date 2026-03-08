import sys
import os
from flask import Flask, render_template_string, request, redirect, url_for
import psycopg2
import threading

# Configuration du path pour trouver le module orchestration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from orchestration.orchestration import traiter_pipeline

app = Flask(__name__)

DB_CONFIG = {
    "host": "127.0.0.1", "database": "pfe_news",
    "user": "postgres", "password": "postgres", "port": "5432"
}

# Template HTML amélioré avec style CSS et gestion de l'état des checkbox
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>PFE News</title></head>
<body>
    <h1>Mes Stories d'Actualités</h1>
    
    <form action="/show-stories" method="post">
        <strong>Choisis tes intérêts :</strong><br>
        {% for interest in all_interests %}
            <input type="checkbox" name="interests" value="{{ interest }}" 
                   {% if interest in selected_interests %}checked{% endif %}> {{ interest }}<br>
        {% endfor %}
        <button type="submit">Générer/Afficher</button>
    </form>

    <hr>
    <form action="/add-interest" method="post">
        <input type="text" name="new_interest" placeholder="Nouvel intérêt...">
        <button type="submit">Ajouter</button>
    </form>

    {% if stories %}
        {% for interest, articles in stories.items() %}
            <div style="border: 1px solid #ccc; padding: 15px; margin: 20px 0; border-radius: 12px; background-color: #f9f9f9;">
                <h3>{{ interest }}</h3>
                {% for art in articles %}
                    <div style="margin-bottom: 15px; border-bottom: 1px solid #eee;">
                        <strong>{{ art[0] }}</strong> <small>({{ art[1] }})</small>
                        <p>{{ art[2] }}</p>
                    </div>
                {% endfor %}
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
    cur.execute("SELECT DISTINCT interest FROM article_summaries;")
    interests = [row[0] for row in cur.fetchall()]
    cur.close(); conn.close()
    return render_template_string(HTML_TEMPLATE, all_interests=interests, selected_interests=[], stories=None)

@app.route('/add-interest', methods=['POST'])
def add_interest():
    topic = request.form.get('new_interest')
    if topic:
        threading.Thread(target=traiter_pipeline, args=(topic,)).start()
    return redirect(url_for('index'))

@app.route('/show-stories', methods=['POST'])
def show_stories():
    selected = request.form.getlist('interests')
    conn, cur = get_db_cursor()
    
    # Récupérer tous les intérêts existants pour le menu
    cur.execute("SELECT DISTINCT interest FROM article_summaries;")
    all_ints = [row[0] for row in cur.fetchall()]
    
    # Récupérer les stories des sélectionnés
    stories = {}
    for interest in selected:
        cur.execute("""SELECT title, published_date, summary_fr FROM article_summaries 
                        WHERE interest = %s AND writing_run_id = 
                        (SELECT MAX(writing_run_id) FROM article_summaries WHERE interest = %s)
                        ORDER BY published_date DESC LIMIT 3;""", (interest, interest))
        stories[interest] = cur.fetchall()
        
    cur.close(); conn.close()
    
    return render_template_string(HTML_TEMPLATE, 
                                  all_interests=all_ints, 
                                  selected_interests=selected, 
                                  stories=stories)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)