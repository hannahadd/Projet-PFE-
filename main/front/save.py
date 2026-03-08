from flask import Flask, render_template_string, request, redirect, url_for
import psycopg2
import sys
import os

# Ajout du chemin vers le dossier orchestration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'orchestration')))
from orchestration import lancer_generation_interet

app = Flask(__name__)

# Configuration DB
DB_CONFIG = {
    "host": "127.0.0.1", "database": "pfe_news",
    "user": "postgres", "password": "postgres", "port": "5432"
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>PFE News Stories</title>
    <style>
        .story-card { border: 1px solid #ccc; padding: 15px; margin: 10px; border-radius: 8px; background-color: #f9f9f9; }
        body { font-family: sans-serif; padding: 20px; }
        .add-interest { margin-top: 20px; padding: 10px; background: #e0e0e0; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Mes Stories d'Actualités</h1>
    <form action="/show-stories" method="post">
        <p>Choisis tes intérêts :</p>
        {% for interest in all_interests %}
            <input type="checkbox" name="interests" value="{{ interest }}" {% if interest in selected %}checked{% endif %}> {{ interest }}<br>
        {% endfor %}
        <br><button type="submit">Générer mes Stories</button>
    </form>

    <div class="add-interest">
        <form action="/add-interest" method="post">
            <input type="text" name="new_interest" placeholder="Nouvel intérêt..." required>
            <button type="submit">Ajouter l'intérêt</button>
        </form>
    </div>

    {% if stories %}
        <hr>
        {% for interest, articles in stories.items() %}
            <div class="story-card">
                <h2>Story : {{ interest }}</h2>
                {% if articles %}
                    {% for art in articles %}
                        <p><strong>{{ art[0] }}</strong> ({{ art[1] }})<br>{{ art[2] }}</p>
                    {% endfor %}
                {% else %}
                    <p><em>Pas d'informations pour aujourd'hui ... Revenez demain !</em></p>
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
    cur.execute("SELECT DISTINCT interest FROM article_summaries;")
    interests = [row[0] for row in cur.fetchall()]
    cur.close(); conn.close()
    return render_template_string(HTML_TEMPLATE, all_interests=interests, stories=None)

@app.route('/add-interest', methods=['POST'])
def add_interest():
    topic = request.form.get('new_interest')
    if topic:
        lancer_generation_interet(topic)
    return redirect(url_for('index'))

@app.route('/show-stories', methods=['POST'])
def show_stories():
    selected = request.form.getlist('interests')
    conn, cur = get_db_cursor()
    cur.execute("SELECT DISTINCT interest FROM article_summaries;")
    all_interests = [row[0] for row in cur.fetchall()]
    
    stories = {}
    for interest in selected:
        query = """
            SELECT title, published_date, summary_fr 
            FROM article_summaries 
            WHERE interest = %s 
            AND writing_run_id = (SELECT MAX(writing_run_id) FROM article_summaries WHERE interest = %s)
            ORDER BY published_date DESC LIMIT 3;
        """
        cur.execute(query, (interest, interest))
        stories[interest] = cur.fetchall()
        
    cur.close(); conn.close()
    return render_template_string(HTML_TEMPLATE, all_interests=all_interests, stories=stories, selected=selected)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)