FROM python:2
RUN pip install scipy
RUN pip install pandas
RUN pip install geopy
RUN pip install matplotlib
RUN mkdir /scripts
RUN mkdir /data
RUN mkdir /objects
WORKDIR /scripts
CMD [ "python", "skyclusters.py" ]

