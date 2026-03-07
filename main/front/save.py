from flask import Flask, render_template_string
import psycopg2

app = Flask(__name__)

# CONFIGURATION (à remplir avec les infos de Yohan)
DB_CONFIG = {
    "host": "127.0.0.1",
    "database": "pfe_news",
    "user": "postgres",
    "password": "postgres",
    "port": "5432"
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>PFE News</title></head>
<body>
    <h1>Ma base de news</h1>
    <form action="/show-data" method="get">
        <button type="submit">Afficher les articles</button>
    </form>
    <hr>
    {% if data %}
        <table border="1">
            <tr>
                <th>Titre</th><th>Date de publication</th><th>Summary FR</th>
            </tr>
            {% for row in data %}
            <tr>
                <td>{{ row[0] }}</td><td>{{ row[1] }}</td><td>{{ row[2] }}</td>
            </tr>
            {% endfor %}
        </table>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/show-data')
def show_data():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    # On récupère juste le titre, la source et le score pour l'exemple
    cur.execute("SELECT title, published_date, summary_fr FROM article_summaries LIMIT 10;")
    articles = cur.fetchall()
    cur.close()
    conn.close()
    return render_template_string(HTML_TEMPLATE, data=articles)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)