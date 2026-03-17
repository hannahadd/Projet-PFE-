import sys
import os
import json
from flask import Flask, render_template, request, redirect, url_for, flash
import psycopg2
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from orchestration.orchestration import traiter_pipeline_complet

app = Flask(__name__)
app.secret_key = "supersecretkey"

DB_CONFIG = {
    "host": "127.0.0.1",
    "database": "pfe_news",
    "user": "postgres",
    "password": "postgres",
    "port": "5432"
}


def get_db_cursor():
    conn = psycopg2.connect(**DB_CONFIG)
    return conn, conn.cursor()


def clean_summary(raw):
    """Nettoie summary_fr : extrait le texte si c'est du JSON."""
    if not raw:
        return ""
    raw = raw.strip()
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
            for v in data.values():
                if isinstance(v, str):
                    return v
        except json.JSONDecodeError:
            raw = raw.replace('{"summary_fr": "', '').replace('"}', '').strip()
    return raw


@app.route('/', methods=['GET'])
def index():
    conn, cur = get_db_cursor()
    cur.execute("""
        SELECT DISTINCT interest
        FROM article_summaries
        WHERE (notes IS NULL OR notes = '')
        ORDER BY interest;
    """)
    interests = [row[0] for row in cur.fetchall()]
    cur.close(); conn.close()
    return render_template('index.html',
                           all_interests=interests,
                           selected_interests=[],
                           stories=None)


@app.route('/add-interest', methods=['POST'])
def add_interest():
    topic = request.form.get('new_interest', '').strip().lower()
    if not topic:
        return redirect(url_for('index'))

    conn, cur = get_db_cursor()
    cur.execute("SELECT COUNT(*) FROM article_summaries WHERE interest = %s;", (topic,))
    exists = cur.fetchone()[0] > 0
    cur.close(); conn.close()

    if exists:
        flash(f"L'intérêt « {topic} » est déjà en base.")
    else:
        flash(f"Pipeline lancé pour « {topic} » — actualise dans quelques instants.")
        threading.Thread(target=traiter_pipeline_complet, args=(topic,), daemon=True).start()

    return redirect(url_for('index'))


@app.route('/show-stories', methods=['POST'])
def show_stories():
    selected = request.form.getlist('interests')

    conn, cur = get_db_cursor()
    cur.execute("""
        SELECT DISTINCT interest
        FROM article_summaries
        WHERE (notes IS NULL OR notes = '')
        ORDER BY interest;
    """)
    all_ints = [row[0] for row in cur.fetchall()]

    stories = {}
    for interest in selected:
        # On récupère TOUS les articles du dernier run, sans limite
        # Si ta table a une colonne url, elle est incluse ; sinon on gère le cas
        try:
            cur.execute("""
                SELECT title, published_date, summary_fr, url
                FROM article_summaries
                WHERE interest = %s
                  AND (notes IS NULL OR notes = '')
                  AND writing_run_id = (
                    SELECT MAX(writing_run_id)
                    FROM article_summaries
                    WHERE interest = %s
                  )
                ORDER BY published_date DESC;
            """, (interest, interest))
        except Exception:
            # Si la colonne url n'existe pas
            conn.rollback()
            cur.execute("""
                SELECT title, published_date, summary_fr, NULL as url
                FROM article_summaries
                WHERE interest = %s
                  AND (notes IS NULL OR notes = '')
                  AND writing_run_id = (
                    SELECT MAX(writing_run_id)
                    FROM article_summaries
                    WHERE interest = %s
                  )
                ORDER BY published_date DESC;
            """, (interest, interest))

        rows = cur.fetchall()
        cleaned = []
        for row in rows:
            title    = row[0]
            pub      = row[1]
            summary  = clean_summary(row[2])
            url      = row[3] if len(row) > 3 else None
            cleaned.append((title, pub, summary, url))

        stories[interest] = cleaned

    cur.close(); conn.close()

    return render_template('index.html',
                           all_interests=all_ints,
                           selected_interests=selected,
                           stories=stories)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)