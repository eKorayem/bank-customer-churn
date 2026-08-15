# Bank Customer Churn - Data Dictionary
**Author:** Eslam Atia
**Last Updated:** August 2026

This document maps the schema of the raw banking datasets used to train the churn prediction model.

### Customer Demographics (`customers.csv`)
* **CustomerId:** Unique identifier for the banking customer.
* **Gender:** Biological sex of the customer (Male/Female).
* **Age:** Customer's age in years.
* **Salary:** Estimated annual income of the customer.
* **LocationId:** Foreign key mapping to the geographic location table.
* **Churned:** Target variable (1 = Customer left the bank, 0 = Customer stayed).

### Financial Behavior (`financials.csv`)
* **CustomerId:** Foreign key mapping to the customer table.
* **Tenure:** Number of years the customer has been with the bank.
* **Balance:** Current account balance in the primary currency.
* **NumProducts:** Number of bank products the customer utilizes (e.g., savings, credit card, loan).
* **HasCreditCard:** Binary flag indicating credit card ownership (1 = Yes, 0 = No).
* **IsActive:** Binary flag indicating active account status based on recent transaction history.

### Geography (`locations.csv`)
* **LocationId:** Unique identifier for the branch region.
* **Geography:** Country of the bank branch (e.g., France, Germany, Spain).