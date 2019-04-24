sed -i '/CMD/c\CMD\ \[\ \"python\"\,\ \"finding_clusters_plots\.py\" \]' Dockerfile
docker build -t finding_clusters .
