from datetime import datetime
from boto.s3.connection import S3Connection
import boto3
from botocore.exceptions import NoCredentialsError
import os


def exportToS3 (incomingMsg, fileName, fileType):
    #status - initiation after function is called
    print('Initiation')
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%Y%m%d-%H%M%S")
    datafileName = timestampStr +'___' + fileName + '.'+ fileType 

    #hardcoded due to not being deployed
    BUCKETNAME = 'bucketName'
    BASEPATH = 'base/path/for/file/'
    S3Key  = 'S3AccessKey' #os.getenv('s3_key')
    S3SKey = 'S3SecretKey' #os.getenv('s3_skey')

    try:
        pathUploaded = BASEPATH + str(datafileName)
        # Status - S3 complete path with datafile name
        print ('writing:' + pathUploaded)

        # Define a connection to S3 using secret keys
        conn = S3Connection(S3Key, S3SKey)
        # Testing connection
        if conn != None:
            #Status - Successful connection
            print("Connected to S3")

        s3 = boto3.resource('s3',region_name='us-east-1', aws_access_key_id=S3Key, aws_secret_access_key=S3SKey)
        s3.Object(BUCKETNAME, pathUploaded).put(Body=incomingMsg)
        #Status - Push file successful
        return("Upload Successful: " + pathUploaded)
    except FileNotFoundError:
        return("The file was not found: " + pathUploaded)
    except NoCredentialsError:
        return("Credentials not available: " + pathUploaded)