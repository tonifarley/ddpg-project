# Readme

**Project Name:** ddpg-project

**Description:** 

## Author: Toni Farley (Github username: https://github.com/tonifarley)

## License: MIT License

## Dependencies: see `requirements.txt`

Getting Started:

    Clone the repository:

git clone https://github.com/tonifarley/ddpg-project.git

    Install dependencies:

npm install

    [Any additional instructions for running the project]

Contributing:

Contributions are welcome! Please see the contributing guidelines: CONTRIBUTING.md for information on how to submit pull requests.

Ownership:

All code within this repository is owned by the listed authors. By contributing, you agree to license your contributions under the chosen license.

Disclaimer:

This software is provided "as is" without warranty of any kind, express or implied. The authors are not responsible for any damages arising from the use of this software.





# ddpg-flex
# toni@prodefined.com

Problems

run

new context

Data

Open source data sets

UC Irvine Machine Learning Repository - https://archive.ics.uci.edu/



Kaggle - https://www.kaggle.com/datasets



 for reinforced learning on Hugging Face - 
https://huggingface.co/datasets?task_categories=task_categories:reinforcement-learning&sort=trending

modify params

"Adding noise to an underconstrained neural network model with a small training dataset can have a regularizing effect and reduce overfitting."

In this implementation, we use OU noise on the environment to reduce overfitting during training toward a more generalized model. It has been suggested that 
Move the noise from the enivronment to the agent. 