import smtplib
import ssl
from email.mime.text import MIMEText


def sendAnEmail(to, subject, body, port = 465):
    """
    Function to send an email

    credits: https://stackabuse.com/how-to-send-emails-with-gmail-using-python/
    """

    gmail_user = 'ignacepelckmans@gmail.com'
    gmail_password = 'efmyqetjornfvimt'

    sent_from = gmail_user
    if to.lower() == 'me': to = 'ignacepelckmans@gmail.com'

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sent_from
    msg['To'] = to

    # Create a secure SSL context
    context = ssl.create_default_context()

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
            print('Connected to the server!')
            server.login(gmail_user, gmail_password)
            print('logged in!')
            server.sendmail(sent_from, to, msg.as_string())
            print('Email sent!')
    except:
        print("woepsie, that didn't work!")
