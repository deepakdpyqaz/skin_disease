# Skin Disease Classification Optmization using MRFO algorithm

## File Structure
1. messages/ => This directory holds the two files *requests.proto* which contains the protobuf specification for the communication and *requests_pb2.py* which contains the compiled python file
2. distributed.py => This file holds the logic of distributed system. It manages the communication between client and server
3. server.py => It holds the logic to run on server. It contains the logic of MRFO, and calls the clients for evaluation of fitness function
4. client.py => It holds the logic to run on clients. It contains the logic of evaluation of fitness function, i.e, training and evaluating a classification model.
5. setup.py => It contains the instructions to download and setup the data files and credentials
6. stats.py => This file holds the logic to send statistics of resource utilization of various clients
7. requirements.txt => It contains the requirements of the software

### Configuration files
- .credentials/ => This directory contains the firebase config, telegram config, and pika config
- settings.json => It contains the path to credentials and data folder

