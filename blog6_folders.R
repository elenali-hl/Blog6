# 1. Create folders
dir.create("code", showWarnings = FALSE)
dir.create("data-raw", showWarnings = FALSE)
dir.create("data", showWarnings = FALSE)
dir.create("results/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("results/tables", recursive = TRUE, showWarnings = FALSE)

# 2. Create Python scripts
file.create("code/01_load_data.py")
file.create("code/02_clean_data.py")
file.create("code/03_sql_matching.py")
file.create("code/04_ige_analysis.py")
file.create("code/05_visualizations.py")

# 3. Create project files
file.create("README.md", showWarnings = FALSE)
file.create("requirements.txt", showWarnings = FALSE)
file.create(".gitignore", showWarnings = FALSE)

# 4. Keep empty folders visible in GitHub
file.create("data/.gitkeep")
file.create("results/figures/.gitkeep")
file.create("results/tables/.gitkeep")

# 5. Check project structure
list.files(recursive = TRUE, all.files = TRUE)






