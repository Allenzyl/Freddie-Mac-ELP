"""
@author: Yilun Zhang
This program can automatically downlowd and unzip all the files (around 14GB) in the Quarterly Single Family Loan-Level Dataset 
provided by Freddie Mac. You may need username and password to login into the database and download the data.
"""

username = ''
password = ''

import requests
import re
import zipfile
import pandas as pd

if __name__ == "__main__":
    session=requests.session()
    
    #Login
    baseurl = "https://freddiemac.embs.com/FLoan/secure/auth.php"
    formdata={'pagename':'download2',
              'username':username,
              'password':password}
    content = session.post(baseurl,formdata)
    
    #Agree Term and Condition
    baseurl2 ="https://freddiemac.embs.com/FLoan/Data/download2.php"
    formdata2={'accept':'Yes',
              'action':'acceptTandC',
              'acceptSubmit':'Continue'}
    content = session.post(baseurl2,formdata2)
    
    #Decode the download page and get download link
    sourcecode = content.text
    r=re.compile(r'<a href.*zip')
    x=re.findall(r,sourcecode)
    
    downloadlist=[]
    for eachitem in x:
        loc1=eachitem.find('title')
        loc2=eachitem.find('>')
        p1=eachitem[8:loc1-1]
        p2=eachitem[loc1+7:loc2-1]
        p3=eachitem[loc2+1:]
        downloadlist.append([p1,p2,p3])
        
    pre="https://freddiemac.embs.com/FLoan/Data/" 
    
    #Download all files
    for eachitem in downloadlist:
        print("Downloading",eachitem[2],eachitem[1])
        url = pre + eachitem[0]
        r = session.get(url) 
        with open("./"+eachitem[2], "wb") as code:
            code.write(r.content)
        print("Successful")
        
    # Extract all files
    inputfileloc="./"
    outputfileloc="./ExtractedDataFile/"

    quarterlist=['Q1','Q2','Q3','Q4']
    yearlist=[str(y) for y in range(1999,2018)]
    filelist=[]
    for y in yearlist:
        for q in quarterlist:
            filelist.append(['historical_data1_'+q+y+'.zip',q+'_'+y])
    
    for eachfile in filelist:
        print("Extracting", eachfile[0])
        with zipfile.ZipFile(inputfileloc+eachfile[0],"r") as zip_ref:
            zip_ref.extractall(outputfileloc+eachfile[1])
            
                
    # Merge all data into one
    wd="./ExtractedDataFile/"
    outdirectory="./WorkingData/"
    
    def FILTER_DELINQUENCY_STATUS(i):
        if i == 'R' or i == 'XX':
            return 2
        elif isinstance(i,int):
            if i>=3:
                return 1
            else:
                return 0
        elif isinstance(i,str):
            if int(i)>=3:
                return 1
            else:
                return 0
        else:
            return -1
        
    quarterlist=['Q1','Q2','Q3','Q4']
    yearlist=[str(y) for y in range(1999,2018)]
    foldernamelist=[]
    filemap={}
    
    for q in quarterlist:
        for y in yearlist:
            foldername=q+'_'+y
            foldernamelist.append(foldername)
            filemap[foldername] = ["historical_data1_"+q+y+".txt","historical_data1_time_"+q+y+".txt"]
    
    
    for eachfolder in foldernamelist:
        print(eachfolder)
        #eachfolder=foldernamelist[0]
        filelist=filemap[eachfolder]
        
        Origination_Data = pd.read_csv(wd+eachfolder+'/'+filelist[0],sep='|',header=None)
        Origination_Data.columns=['FICO','First Payment Date','First Time Homebuyer Flag','Maturity Date','MSA','MI%','Units','Occupancy',
                    'CLTV','DTI','OUPB','OLTV','OInterest rate','Channel','PPM','Product Type','Property State','Property Type',
                    'Postal','Loanid','Purpose','OLoan Term','Number of borrowers','Seller','Service','Conforming']
        
        ModifiedOD = Origination_Data[['FICO','First Payment Date','First Time Homebuyer Flag','MSA','Units','Occupancy','CLTV','DTI',
                    'OUPB','OLTV','OInterest rate','Channel','Property State','Property Type','Loanid','Purpose','Number of borrowers']]
        del(Origination_Data)
    
        Monthly_Performance_Data = pd.read_csv (wd+eachfolder+'/'+filelist[1],sep='|',header=None)
        Monthly_Performance_Data.columns=[str(i) for i in range(1,27)]
        
        ModifiedMPD = Monthly_Performance_Data[[str(i) for i in[1,4,5]]]
        del(Monthly_Performance_Data)
    
        
        temp=ModifiedMPD['4'].apply(FILTER_DELINQUENCY_STATUS)
        ModifiedMPD['4']=list(temp)
        del(temp)
        
        min_loan_age=pd.DataFrame(ModifiedMPD['5'].groupby(ModifiedMPD['1']).min())
        max_delinquency_status=pd.DataFrame(ModifiedMPD['4'].groupby(ModifiedMPD['1']).max())
        
        min_loan_age['Loanid']=min_loan_age.index
        min_loan_age.reset_index(drop = True)
        max_delinquency_status['Loanid']=max_delinquency_status.index
        max_delinquency_status=max_delinquency_status.reset_index(drop = True)
        
        MPDfinaldf=pd.merge(min_loan_age,max_delinquency_status,how='left')
        MPDfinaldf=MPDfinaldf.rename(columns={'5':'FirstLoanAge','4':'Dstatus'})
        
        del(min_loan_age)
        del(max_delinquency_status)
        del(ModifiedMPD)
        
        ModifiedOD=ModifiedOD.rename(columns={'20':'Loanid'})
        Alldatafinaldf=pd.merge(ModifiedOD,MPDfinaldf,how='left')
        Alldatafinaldf.to_csv(outdirectory+eachfolder+'.txt',index=False,sep='|',header = None)
        del(ModifiedOD)
        del(MPDfinaldf)
        del(Alldatafinaldf)
