-- Q1: Which customer profiles have the highest churn risk based on gender

WITH
    MainTbl as
    (
    SELECT 
        [Gender],
        count(*) as TotalCustomer,
        sum(CAST([Churned] as int)) as TotalChurn
    FROM [BankChurn].[dbo].[demographic]
    GROUP BY [Gender]
    )

SELECT *,
   FORMAT((TotalChurn*100/TotalCustomer), 'N2')+'%' as ChurnRate
FROM MainTbl


-- Q2: How does churn rate vary across customer segments
USE [BankChurn];
WITH MainTbl AS(
    SELECT 
        case
            when D.[Age] < 30 then 'Less than 30'
            when D.[Age] between 30 and 50 then 'Between 30-50'
            else 'Above 50'
        END AS AgeGrp,
        D.[Churned],
        L.[Geography] as Country
    FROM [dbo].[demographic] D
    JOIN [dbo].[location] L ON L.[LocationId] = D.[LocationId]),

    SecondTbl as (
        SELECT
            Country, AgeGrp,
            COUNT(*) AS TotalCustomer, 
            AVG(CAST(Churned as float)) as AvgChurnRate,
            AVG(AVG(CAST(Churned as float))) OVER(partition by Country) as AvgChurnCountry
        FROM MainTbl
        GROUP BY Country, AgeGrp
    )
SELECT *,
    AvgChurnCountry - AvgChurnRate
FROM SecondTbl


-- Q3: How does churn behavior change when we dynamically sclice customers by buisness paramaters

USE [BankChurn];
GO

DECLARE  @MinTenure INT = 2;
DECLARE @MaxBalance DECIMAL = 70000;
DECLARE @MaxProduct INT = 7

SELECT
    A.[CustomerId],
    A.[Tenure],
    A.[Balance],
    A.[NumProducts],
    D.[Churned]
FROM
    [dbo].[account] A
JOIN
    [dbo].[demographic] D
ON 
    D.[CustomerId] = A.[CustomerId]
WHERE
    [Tenure] > @MinTenure
    AND [Balance] < @MaxBalance
    AND [NumProducts] < @MaxProduct