from flask import Flask, render_template_string, request
import psycopg2

app = Flask(__name__)

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
        .story-card { border: 1px solid #ccc; padding: 15px; margin: 10px; border-radius: 8px; }
        body { font-family: sans-serif; padding: 20px; }
    </style>
</head>
<body>
    <h1>Mes Stories d'Actualités</h1>
    <form action="/show-stories" method="post">
        <p>Choisis tes intérêts :</p>
        {% for interest in all_interests %}
            <input type="checkbox" name="interests" value="{{ interest }}"> {{ interest }}<br>
        {% endfor %}
        <br><button type="submit">Générer mes Stories</button>
    </form>

    {% if stories %}
        <hr>
        {% for interest, articles in stories.items() %}
            <div class="story-card">
                <h2>Story : {{ interest }}</h2>
                {% for art in articles %}
                    <p><strong>{{ art[0] }}</strong> ({{ art[1] }})<br>{{ art[2] }}</p>
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
    return render_template_string(HTML_TEMPLATE, all_interests=interests)

@app.route('/show-stories', methods=['POST'])
def show_stories():
    selected = request.form.getlist('interests')
    conn, cur = get_db_cursor()
    
    cur.execute("SELECT DISTINCT interest FROM article_summaries;")
    all_interests = [row[0] for row in cur.fetchall()]
    
    stories = {}
    for interest in selected:
        cur.execute("SELECT title, published_date, summary_fr FROM article_summaries WHERE interest = %s LIMIT 3;", (interest,))
        stories[interest] = cur.fetchall()
        
    cur.close(); conn.close()
    return render_template_string(HTML_TEMPLATE, all_interests=all_interests, stories=stories)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)