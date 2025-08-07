# Deep Learning with PyTorch - Session 10: Version Control & Progressive Model Building  
  
**Objective:**    
Learn version control through implementing a classic PyTorch pipeline, committing each step, and pushing to a remote repository via GitHub Desktop.  
  
---  
  
## Session Timeline (2 Hours)  
  
| Time      | Activity                                       |  
| --------- | ---------------------------------------------- |  
| 0:00-0:10 | 1. Check-in + Recap last session               |  
| 0:10-0:20 | 2. What is Version Control? Why use Git?        |  
| 0:20-0:45 | 3. Setting Up Your GitHub & Project Repo        |  
| 0:45-1:40 | 4. Guided Solo Exercise (with stepwise commits) |  
| 1:40-1:55 | 5. Review Visualizations & Final Commit         |  
| 1:55-2:00 | 6. Discussion & Q/A, Wrap-up                    |  
  
---  
  
## Session Steps & Instructions  
  
### 1. **Intro: Recap & Version Control Motivation** (10 min)  
- Welcome, attendance, brief recap.  
- Explain git basics: commits, branches, tracking changes; why versioning is essential for code and experiments.  
  
---  
  
### 2. **GitHub & GitHub Desktop Setup** (10 min)  
1. Sign up for a GitHub account if needed.  
2. Download and install GitHub Desktop: https://desktop.github.com/  
3. Sign in to GitHub via GitHub Desktop.  
  
---  
  
### 3. **Create a New Repo for Your Project** (15 min)  
- In GitHub Desktop: `File > New Repository`  
- Name: `pytorch-solo-exercise`  
- Add `.gitignore`: Python  
- Add a README if prompted.  
- Click `Create Repository`.  
- Publish to GitHub (top navigation in GitHub Desktop).  
  
**Result:** Your new project exists both locally and on GitHub.  
  
---  
  
### 4. **Add the Exercise Script & Make Initial Commit** (5 min)  
- Use the provided `solo_exercise.py` script with TODO sections such as:  
    - Load dataset (e.g., MNIST, Iris, etc.)  
    - Split data  
    - Define model  
    - Loss function & optimizer  
    - Training loop  
    - Evaluation  
    - Plotting code present but non-functional until previous TODOs are filled  
- Place the script in your repo folder.  
- In GitHub Desktop, you’ll see uncommitted changes.  
- Commit message: “Initial commit: add solo_exercise.py with TODOs”  
- Click “Commit to main”.  
- Click “Push origin”.  
  
---  
  
### 5. **Progressive Exercise & Feature Commits** (60 min)  
Fill in the TODOs step by step, committing and pushing each time:  
  
1. **Data Loading**  
    - Complete dataset loading code.  
    - Commit: “Add data loading code”  
    - Push.  
  
2. **Train/Test Split**  
    - Implement train/test split.  
    - Commit: “Implement dataset split into train/test”  
    - Push.  
  
3. **Model Definition**  
    - Define a simple neural network.  
    - Commit: “Define model architecture”  
    - Push.  
  
4. **Loss Function & Optimizer**  
    - Specify loss and optimizer.  
    - Commit: “Add loss function and optimizer”  
    - Push.  
  
5. **Training Loop**  
    - Implement the training loop.  
    - Commit: “Add training loop”  
    - Push.  
  
6. **Evaluation**  
    - Add code to evaluate your model’s performance.  
    - Commit: “Add evaluation code”  
    - Push.  
  
7. **Plotting**  
    - Make plotting code functional (e.g., loss curves/accuracy).  
    - Commit: “Enable plotting of results”  
    - Push.  
  
---  
  
### 6. **Reviewing Outputs & Finalizing** (15 min)  
- Ensure visualizations (e.g., `loss_curve.png`) are generated and saved.  
- In GitHub Desktop: add new files.  
- Commit: “Add training visualizations”  
- Push.  
- (Optional) Add a short `results.md` summary and commit.  
  
---  
  
### 7. **Discussion & Wrap-Up** (5 min)  
- Recap: benefits of version control and regular commits in ML projects.  
- (Optional) Invite students to share GitHub links.  
  
---  
  
## **Supporting Materials**  
- `solo_exercise.py` template (with "# TODO:" comments)   
- Cheat sheet: [GitHub Desktop Shortcuts](https://docs.github.com/en/desktop/overview/getting-started-with-github-desktop)  
  