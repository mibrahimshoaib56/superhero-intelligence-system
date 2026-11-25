import nbformat
from nbclient import NotebookClient

with open("superhero_project.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name="superhero_env")
print("Executing superhero_project.ipynb with superhero_env...")
client.execute()

with open("superhero_project.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Execution completed successfully! All cell outputs and plots saved.")
