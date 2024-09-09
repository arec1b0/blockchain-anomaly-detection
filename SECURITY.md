# Security Policy

Hello, I’m Daniil Krizhanonovskyi, and I take security very seriously. The **Blockchain Anomaly Detection** project is intended to help detect anomalies in blockchain transactions, and it's crucial that this tool remains secure and reliable for everyone who uses it.

This document outlines the security policy for the **Blockchain Anomaly Detection** project and provides instructions for reporting vulnerabilities.

## Supported Versions

The following versions of the **Blockchain Anomaly Detection** project are actively maintained and supported:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark:  |
| < 1.0   | :x:                |

I encourage all users to upgrade to the latest stable version of the project to ensure they are protected by the most recent security updates and improvements.

## Reporting a Vulnerability

If you discover a vulnerability in this project, I ask that you **do not report it publicly**. Instead, please follow the steps outlined below to report the vulnerability directly to me. I will make every effort to address the issue as quickly as possible.

### How to Report

1. **Email me directly** at **[daniil.krizhanovskyi@hotmail.com]** with the subject line: `[Security Issue] Blockchain Anomaly Detection`.
2. In your report, please include the following details:
   - A description of the vulnerability and its potential impact.
   - The steps to reproduce the issue.
   - Any code or data that demonstrates the vulnerability.
   - Any additional context or details that might help in understanding the issue.
   
   If possible, please provide any mitigation strategies you might have used to address the issue in your local environment.

3. **Expect a response**: I will acknowledge receipt of your report within 48 hours and will strive to investigate and address the issue within a week. I will keep you updated on the progress and may ask for additional information or clarifications if needed.

4. **Coordinated disclosure**: If the issue is confirmed and a fix is implemented, I will release a security patch and work with you to coordinate the disclosure of the vulnerability in a responsible manner.

## Security Measures

To minimize security risks and protect the integrity of the **Blockchain Anomaly Detection** project, I follow best practices throughout the development process:

- **Secure Coding Practices**: The project follows secure coding guidelines to ensure the integrity of the codebase. This includes input validation, avoiding dangerous or outdated dependencies, and minimizing attack vectors.
  
- **API Security**: The integration with the Etherscan API is secured through API keys, which are never hardcoded in the source code. All API keys are stored in environment variables, and users are encouraged to rotate their keys regularly.

- **Regular Updates**: I will regularly review and update project dependencies to address any vulnerabilities that may be identified in third-party libraries.

- **Code Reviews**: All new code is subject to thorough reviews to ensure that it does not introduce security vulnerabilities.

- **Logging and Monitoring**: Detailed logging and monitoring have been implemented across the project, allowing users to track the performance and security of the application. However, sensitive data is never logged.

## Best Practices for Users

To further ensure the security of your setup while using the **Blockchain Anomaly Detection** tool, I recommend the following best practices:

1. **Use the Latest Version**: Always ensure that you are running the latest version of the project to benefit from the most recent security patches and improvements.

2. **Secure API Keys**: Always store your Etherscan API keys in secure environment variables, and do not share them publicly or hardcode them into your code.

3. **Monitor Logs**: Regularly monitor the application logs for any unusual activity. The project’s logging mechanism is designed to give users visibility into any potential issues.

4. **Limit Access**: If deploying the tool in a production environment, ensure that access is restricted to trusted users. Use firewalls, VPNs, and other access control mechanisms to limit who can interact with the system.

5. **Report Issues**: If you encounter any bugs, issues, or security concerns, please don’t hesitate to reach out. Even minor issues can sometimes lead to larger vulnerabilities.

## Vulnerability Disclosure

Once a vulnerability is reported and a fix is implemented, I will release a new version of the project containing the security patch. Users will be notified through the project’s GitHub releases and changelog. In cases where a vulnerability is publicly disclosed, I will provide a detailed post-mortem report on the issue, including its impact, resolution, and any steps users need to take.

---

I’m committed to maintaining a secure project, and I truly appreciate your help in making **Blockchain Anomaly Detection** as safe and reliable as possible. If you have any questions about this security policy or need further assistance, feel free to reach out to me directly.

Thank you for your support!

Best regards,  
**Daniil Krizhanonovskyi**
