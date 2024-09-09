# Contributing to Blockchain Anomaly Detection

Hello, and thank you for your interest in contributing to the **Blockchain Anomaly Detection** project. My name is Daniil Krizhanonovskyi, and I’m the creator of this project. I truly appreciate the time and effort you're willing to dedicate to improving this project. To make sure that the contribution process is smooth and collaborative, I’ve outlined some guidelines and instructions below.

## How Can You Contribute?

There are many ways you can contribute to this project:
1. **Reporting Bugs**: If you find any bugs or issues, please report them.
2. **Suggesting Features**: I’m always open to new ideas for features or improvements.
3. **Contributing Code**: If you're a developer, you can contribute directly by fixing bugs, adding new features, or improving existing code.
4. **Improving Documentation**: Documentation is key for any open-source project. If you notice missing or incomplete documentation, feel free to improve it.

## How to Contribute Code

If you'd like to contribute code, here are the steps to get started:

### Step 1: Fork the Repository

To contribute to the project, first, you need to fork the repository on GitHub. This will create a personal copy of the repository under your GitHub account.

- Go to the project’s repository:  
  `https://github.com/dkrizhanovskyi/blockchain-anomaly-detection`
- Click on the “Fork” button in the top-right corner of the page.

### Step 2: Clone Your Fork

Once you have forked the repository, clone your fork to your local machine using the following command:

```bash
git clone https://github.com/dkrizhanovskyi/blockchain-anomaly-detection.git
```

This will download your personal copy of the repository.

### Step 3: Create a New Branch

Before making any changes, create a new branch for your contribution. This helps to keep your contributions separate from the main project code and makes it easier to manage multiple features or fixes.

```bash
git checkout -b feature-branch-name
```

Replace `feature-branch-name` with a meaningful name that describes what you are working on, for example, `fix-api-bug` or `add-new-feature`.

### Step 4: Make Your Changes

Now that you are on your new branch, make the necessary changes to the code or documentation. Please ensure that your changes are in line with the project's existing coding style and follow the guidelines mentioned below.

### Step 5: Write Tests

If you’re making any code changes, especially adding new features or fixing bugs, please make sure to write unit tests for your changes. The project uses `pytest` for testing, and tests are located in the `tests/` directory. You can run the tests using the following command:

```bash
pytest
```

### Step 6: Commit and Push Your Changes

After making your changes and verifying that everything works correctly (including the tests), commit your changes with a clear and descriptive commit message. Make sure to follow the conventional commit style.

```bash
git commit -m "fix(api): resolve API timeout issue"
```

Once committed, push your changes to your forked repository:

```bash
git push origin feature-branch-name
```

### Step 7: Create a Pull Request

After pushing your changes, navigate to the original repository and create a pull request (PR) from your fork. The pull request should clearly describe the changes you've made and reference any related issues if applicable.

- Go to `https://github.com/dkrizhanovskyi/blockchain-anomaly-detection`
- Click on the **Pull Requests** tab, and then click **New Pull Request**.
- Select the branch you just created and compare it to the `main` branch of the project.
- Click on **Create Pull Request**.

I will review your pull request as soon as possible. Please be patient, as it may take some time to thoroughly review the code. If any changes are requested during the review, I’ll leave comments, and you can address them by pushing new commits to your branch.

### Step 8: Respond to Feedback

If I provide any feedback or requests for changes, please address those in your pull request. Update your branch, and once you're ready, push your changes. This will automatically update the pull request.

## Style Guidelines

To keep the code clean and readable, I follow specific coding standards. Please make sure your contributions adhere to the following guidelines:

1. **Follow PEP 8**: For Python code, adhere to the PEP 8 style guide for writing clean and readable code.
2. **Write Docstrings**: Each function, class, and module should have clear docstrings that describe what it does.
3. **Write Tests**: Always accompany your code with unit tests.
4. **Commit Messages**: Use clear and concise commit messages, following the conventional commit style. For example:
   - `fix(api): resolve API timeout issue`
   - `feat(detection): add new anomaly detection algorithm`
   - `docs(readme): update installation instructions`

## Bug Reporting

If you encounter a bug, you can report it by opening an issue on the GitHub repository. Please include the following details in your bug report:

1. A clear and concise description of the issue.
2. Steps to reproduce the bug.
3. Expected behavior vs actual behavior.
4. Screenshots or logs, if applicable.
5. Any additional context that might help in debugging the issue.

To report a bug:
- Go to the **Issues** tab on the repository.
- Click **New Issue**.
- Choose **Bug Report** and fill out the template.

## Suggesting Features

If you have an idea for a new feature, feel free to suggest it by opening a new issue on the repository. Describe the feature in detail, including how it can be implemented and what benefits it will bring to the project.

To suggest a feature:
- Go to the **Issues** tab on the repository.
- Click **New Issue**.
- Choose **Feature Request** and fill out the template.

## Code of Conduct

This project follows a [Code of Conduct](CODE_OF_CONDUCT.md), and I expect all contributors to adhere to it. Please make sure you are respectful to others, provide constructive feedback, and collaborate in a positive and inclusive manner.

## Security Issues

If you discover a security vulnerability, please do not report it via a public issue. Instead, email me directly at [daniill.krizhanovskyi@hotmail.com]. I take security vulnerabilities seriously and will address them as soon as possible.

## Thank You!

Thank you again for your interest in contributing to the **Blockchain Anomaly Detection** project. Your contributions are what make open-source projects like this possible. I’m excited to see how we can improve this project together and create something truly valuable for the community.

Best regards,  
**Daniil Krizhanonovskyi**
