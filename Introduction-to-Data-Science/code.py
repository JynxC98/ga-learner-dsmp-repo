# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 



# code starts here
df = pd.read_csv(path)
bank = pd.DataFrame(df)
bank.head()
categorical_var = bank.select_dtypes(include = 'object')
print(categorical_var)
numerical_var = bank.select_dtypes(include = 'number')
print(numerical_var)




# code ends here


# --------------
# code starts here
banks= bank.drop(columns='Loan_ID')
print(banks.isnull().sum())

# apply mode 

bank_mode = banks.mode().iloc[0]

# Fill the missing values with 

banks.fillna(bank_mode, inplace=True)

# check again all the missing values filled.

print(banks.isnull().sum())





# --------------
avg_loan_amount = banks.pivot_table(values=["LoanAmount"], index=["Gender","Married","Self_Employed"], aggfunc=np.mean)


print (avg_loan_amount)
# code ends here



# --------------
# code starts here
loan_approved_se = banks[(banks['Self_Employed']=='Yes') & (banks['Loan_Status']=='Y')]
loan_approved_nse = banks[(banks['Self_Employed']=='No') & (banks['Loan_Status']=='Y')]
percentage_se = (len(loan_approved_se)/(len(banks['Loan_Status'])))*100
percentage_nse = (len(loan_approved_nse)/(len(banks['Loan_Status'])))*100
# code ends here


# --------------
# code starts here

loan_term = banks['Loan_Amount_Term'].apply(lambda x: int(x)/12 )


big_loan_term=len(loan_term[loan_term>=25])

print(big_loan_term)
# code ends here


# --------------
# code starts here
import pandas as pd
loan_groupby = banks.groupby(['Loan_Status'])
loan_groupby = loan_groupby['ApplicantIncome', 'Credit_History']

mean_values = loan_groupby.mean()



# code ends here


