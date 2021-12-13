FROM python:3.8-slim-buster

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install wget cabextract xfonts-utils && \
    wget http://ftp.de.debian.org/debian/pool/contrib/m/msttcorefonts/ttf-mscorefonts-installer_3.8_all.deb && \
    dpkg -i ttf-mscorefonts-installer_3.8_all.deb && \
    apt-get -y install fonts-takao \
                       fonts-ipafont \
                       fonts-ipaexfont
                       
WORKDIR /code
COPY requirements.txt /code

RUN pip3 install -r requirements.txt
ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]
