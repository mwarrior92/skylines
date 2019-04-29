FROM python:2
RUN pip install scipy
RUN pip install pandas
RUN pip install geopy
RUN pip install matplotlib
RUN pip install IPy
RUN pip install statsmodels
RUN mkdir /scripts
RUN mkdir /data
RUN mkdir /objects
RUN mkdir /plots
WORKDIR /scripts
RUN mkdir -p /root/.config/matplotlib
COPY matplotlibrc /root/.config/matplotlib/matplotlibrc
RUN echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc
RUN pip install -U scikit-learn
RUN pip install geopandas
RUN pip install descartes
CMD [ "python", "finding_clusters_plots.py" ]

