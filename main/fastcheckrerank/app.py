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
                <th>Interest</th><th>Titre</th><th>Summary FR</th><th>Notes</th>
            </tr>
            {% for row in data %}
            <tr>
                <td>{{ row[0] }}</td><td>{{ row[1] }}</td><td>{{ row[2] }}</td><td>{{ row[3] }}</td>
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
    cur.execute(
        """
        SELECT interest, title, summary_fr, notes
        FROM article_summaries
        WHERE writing_run_id = (SELECT MAX(id) FROM writing_runs)
        ORDER BY interest ASC, rank ASC;
        """
    )
    articles = cur.fetchall()
    cur.close()
    conn.close()
    return render_template_string(HTML_TEMPLATE, data=articles)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8088)