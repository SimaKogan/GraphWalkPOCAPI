import markdown
import webbrowser
import tempfile
import os

def display_hebrew_markdown_pycharm(md_text):
    # Convert Markdown to HTML
    html_body = markdown.markdown(md_text)
    html = f"""
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{ direction: rtl; font-size: 18px; font-family: 'Arial', sans-serif; }}
      </style>
    </head>
    <body>
      {html_body}
    </body>
    </html>
    """

    # Write HTML to a temporary file and open in browser
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html', encoding='utf-8') as f:
        f.write(html)
        temp_html_path = f.name

    file_url = 'file://' + os.path.realpath(temp_html_path)
    print(f"\n🔗 Preview your Markdown here:\n{file_url}\n")
    webbrowser.open(file_url)
