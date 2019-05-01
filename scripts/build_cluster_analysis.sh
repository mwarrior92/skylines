sed -i '/CMD/c\CMD\ \[\ \"python\"\,\ \"section_cluster_analysis\.py\" \]' Dockerfile
docker build -t cluster_analysis .

