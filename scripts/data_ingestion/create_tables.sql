

-- Create demographic table --
CREATE TABLE demographic(
    CustomerId INT primary key IDENTITY(1, 1),
    Gender nvarchar(10),
    Age INT,
    Salary DECIMAL(10, 2),
    LocationId INT,
    Churned BIT
)




-- Create Account table --
CREATE TABLE account(
    CustomerId int primary key IDENTITY(1, 1),
    Tenure int,
    Balance DECIMAL(10, 2),
    NumProducts int,
    HasCreditCard bit,
    IsActive bit

);


-- Create Location table --
CREATE TABLE location(
    LocationId int primary key IDENTITY(1, 1),
    [Geography] NVARCHAR(15)
);

