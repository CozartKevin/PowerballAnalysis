# Powerball Analysis

**Hands-on tinkering with real Powerball data to learn Python workflows and analysis basics.**

## Goals
- Practice Python on a real dataset
- Explore basic data cleaning and workflow design
- Try out simple analysis and visualization
- Experiment with frequency-weighted combinations

## Tools & Libraries
- Python
- pandas (data handling)
- matplotlib / seaborn (visualization)
- NumPy (basic calculations)
- itertools, random (combinatorial sampling)

---

## Core Analysis (Implemented by Author, Assisted by ChatGPT)

This section contains the analysis and visualizations implemented by the author. Guidance, code suggestions, and workflow assistance were provided by ChatGPT.

### What This Section Does
- Loads Powerball historical data (post-October 2015)
- Cleans and organizes the dataset
- Runs statistical summaries (frequency counts, averages, distributions)
- Visualizes data with modular charts:
  - Individual ball frequencies
  - All main balls combined
  - Sum of main balls
  - Even/Odd distribution per draw
  - High/Low distribution per draw
  - Top number highlights

### Key Insights
- Most frequent numbers per ball and overall
- Numbers never drawn
- Distribution patterns in sums, even/odd counts, and high/low counts
- Visualization of correlations among main balls and with Powerball

**Charts generated include:**  
- Per-ball number frequencies  
- Overall main ball frequencies  
- Correlation matrices  
- Even/Odd and High/Low distributions  
- Top 5 Powerball and top 20 main ball numbers

---

## ChatGPT-Enhanced Predictive Section (Optional / “Show-Off”)

This section demonstrates a more advanced approach suggested by ChatGPT. It uses historical data to generate weighted combinations of numbers that align with observed distributions. **This section is not authored solely by the project author.**

### Features
- Generates a large pool of 300 combinations using:
  - Frequency weights (main and Powerball numbers)
  - Co-occurrence of pairs and triples
  - Even/Odd and High/Low distributions
  - Sum-of-main-balls adjustment
- Applies top-number multipliers to emphasize historically common numbers
- Scores each combination and keeps the top 10
- Visualizes top combinations using heatmaps
- Saves results to CSV and PNG files

**Outputs include:**  
- Top 10 weighted combinations  
- Heatmap visualization of combinations  
- CSV of suggested top combinations

---

## Notes
This is a **curiosity-driven project**, not intended as a polished data science portfolio piece. The focus is on learning Python workflows, exploring combinatorial logic, and experimenting with real-world data.

## Acknowledgements
- [Data obtained from Data.gov](https://catalog.data.gov/dataset/lottery-powerball-winning-numbers-beginning-2010)  
- Guidance and code suggestions provided by ChatGPT
