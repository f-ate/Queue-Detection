# Queue-Detection

# Queue Detection on Yolov5 using Jetson Nano 2gb Developer Kit.

Queue detection system which will detect a group of people in a single file or line
and if not in single line it will notify on the viewfinder.
## Aim and Objectives

### Aim

To create a Queue detection system which will detect a group of people in a single file or line
and if not in single line it will notify on the viewfinder.

### Objectives

• The main objective of the project is to create a program which can be either run on Jetson
nano or any pc with YOLOv5 installed and start detecting using the camera module on the
device.

• Using appropriate datasets for recognizing and interpreting data using machine learning.

• To show on the optical viewfinder of the camera module whether the group of people are in
line or not.
## Abstract

• Group of people are classified based on whether they are in line or not and is detected by
the live feed from the system’s camera.

• We have completed this project on jetson nano which is a very small computational device.

• A lot of research is being conducted in the field of Computer Vision and Machine Learning
(ML), where machines are trained to identify various objects from one another. Machine
Learning provides various techniques through which various objects can be detected.

• One such technique is to use YOLOv5 with Roboflow model, which generates a small
size trained model and makes ML integration easier.

• Queue areas are places in which people queue for goods or services. Such a group of
people is known as a queue or line, and the people are said to be waiting or standing in a
queue or in line, respectively.

• Queuing is governed as much by environmental design as unspoken rules promoting
equality and efficiency.
## Introduction

• This project is based on a Queue detection model with modifications. We are going to
implement this project with Machine Learning and this project can be even run on jetson
nano which we have done.

• This project can also be used to gather information about whether the group of people are
in Queue or not.

• If someone in Queue breaks the line and tries to come forward the model can even be
trained to check the colour of clothes a person is wearing based on the image annotations,
we give in roboflow.

• Queue detection sometimes become difficult as people don’t usually stand in straight line
they sometimes stand from shoulder to shoulder and might even take a turnabout in order
to talk to their friends behind them. However, training in Roboflow has allowed us to crop
images and also change the contrast of certain images to match the time of day for better
recognition by the model. Also, many more images can be added in roboflow to increase
the accuracy of the model.

• Neural networks and machine learning have been used for these tasks and have obtained
good results.

• Machine learning algorithms have proven to be very useful in pattern recognition and
classification, and hence can be used for Queue detection as well.
## Literature Review

• Queues can be found in railway stations to book tickets, at bus stops for boarding and at
temples as well. Queues are generally found at transportation terminals where security
screenings are conducted.

• Large stores and supermarkets may have dozens of separate queues, but this can cause
frustration, as different lines tend to be handled at different speeds; some people are served
quickly, while others may wait for longer periods of time.

• Professor Nick Haslam, from the Melbourne School of Psychological Sciences at the
University of Melbourne, describes queuing as a social norm, governed by unspoken rules
promoting efficiency and equality. “Queuing exists because there is an imbalance between
the supply and demand of services,” he says. “If we could get the desired level of service
when we wanted it, there wouldn’t be queues.

• But in a world where there is more demand than supply, queuing is a very efficient way
to deliver a service without having a scrum of people fighting to get to it first. “It also
prevents people who are the loudest, the most devious, the most assertive or the biggest
from gaining unfair advantage.” With many services in constant demand, queuing is
inevitable. In fact, Professor Haslam says most service providers actively encourage their
customers to queue, all without saying a word.

• “If you see one of those long, serpentine queues leading to the Qantas check-in counters,
it is very clear that you are supposed to queue,” says Professor Haslam. “The environment
is set up to imply queuing, and people are accustomed to following those expectations.”

• “People usually choose to queue because it is fair,” Professor Haslam says. “In fact, queues
are places where people are obsessed with fairness, and were cutting in line is seen as a
terrible crime that can lead to all sorts of scuffles, fights and frictions. “Ultimately, queuing
defines a clear relationship between when you arrive and when you receive the service
you need. People find that satisfying.

• “If there are parallel queues, people tend to think the other queues are moving faster,”
Professor Haslam adds. “We’re very, very alert. When you queue, the whole issue of
fairness is so salient in your mind that you compare yourself implicitly to the people next
to you. And people become quite unhappy if other queues move faster.”

• While our approach to queuing may seem inflexible, there are degrees of fairness – and
Professor Haslam says it depends on the situation. “In emergency rooms, for instance, we
don’t expect orderly queues. People generally don’t mind being bumped by someone who
comes in with a more severe condition than they have. So, if you’re waiting to have a little
cut on your hand treated, and someone comes in with a broken leg, we don’t mind
violations of queuing order in those circumstances.”
## Jetson Nano Compatibility

• The power of modern AI is now available for makers, learners, and embedded developers
everywhere.

• NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run
multiple neural networks in parallel for applications like image classification, object
detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as
little as 5 watts.

• Hence due to ease of process as well as reduced cost of implementation we have used Jetson
nano for model detection and training.

• NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated
AI applications. All Jetson modules and developer kits are supported by JetPack SDK.

• In our model we have used JetPack version 4.6 which is the latest production release and
supports all Jetson modules.
## Proposed System

1. Study basics of machine learning and image recognition.
    
2. Start with implementation
        
        ➢ Front-end development
        ➢ Back-end development

3. Testing, analysing and improvising the model. An application using python and Roboflow and its machine learning libraries will be using machine learning to identify whether group of people are in Queue or not.

4. Use datasets to interpret the object and suggest whether the group on the camera’s viewfinder is in Queue or not.
## Methodology

The Queue detection system is a program that focuses on implementing real time Queue
detection.

It is a prototype of a new product that comprises of the main module:

Queue detection and then showing on viewfinder whether the group of people are in Queue
or not.

Queue Detection Module

```bash
This Module is divided into two parts:
```

    1] Group Detection

➢ Ability to detect the location of a group of people in any input image or frame. The
output is the bounding box coordinates on the detected group.

➢ For this task, initially the Dataset library Kaggle was considered. But integrating
it was a complex task so then we just downloaded the images from gettyimages.ae
and google images and made our own dataset.

➢ This Dataset identifies group of people on a Bitmap graphic object and returns the
bounding box image with annotation of whether in Queue or not present in a given
image.

    2] Queue Detection

➢ Recognition of the group of people and whether they are in Queue or not.

➢ Hence YOLOv5 which is a model library from roboflow for image classification
and vision was used.

➢ There are other models as well but YOLOv5 is smaller and generally easier to use
in production. Given it is natively implemented in PyTorch (rather than Darknet),
modifying the architecture and exporting and deployment to many environments
is straightforward.

➢ YOLOv5 was used to train and test our model for whether the group of people are
in Queue or not. We trained it for 149 epochs and achieved an accuracy of
approximately 92%.
## Installation

### Initial Setup

Remove unwanted Applications.
```bash
sudo apt-get remove --purge libreoffice*
sudo apt-get remove --purge thunderbird*
```
### Create Swap file

```bash
sudo fallocate -l 10.0G /swapfile1
sudo chmod 600 /swapfile1
sudo mkswap /swapfile1
sudo vim /etc/fstab
```
```bash
#################add line###########
/swapfile1 swap swap defaults 0 0
```
### Cuda Configuration

```bash
vim ~/.bashrc
```
```bash
#############add line #############
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export
LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_P
ATH}}
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
```
```bash
source ~/.bashrc
```
### Udpade a System
```bash
sudo apt-get update && sudo apt-get upgrade
```
################pip-21.3.1 setuptools-59.6.0 wheel-0.37.1#############################

```bash 
sudo apt install curl
```
``` bash 
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
```
``` bash
sudo python3 get-pip.py
```
```bash
sudo apt-get install libopenblas-base libopenmpi-dev
```
```bash
sudo apt-get install python3-dev build-essential autoconf libtool pkg-config python-opengl
python-pil python-pyrex python-pyside.qtopengl idle-python2.7 qt4-dev-tools qt4-designer
libqtgui4 libqtcore4 libqt4-xml libqt4-test libqt4-script libqt4-network libqt4-dbus python-
qt4 python-qt4-gl libgle3 python-dev libssl-dev libpq-dev python-dev libxml2-dev libxslt1-
dev libldap2-dev libsasl2-dev libffi-dev libfreetype6-dev python3-dev
```
```bash
vim ~/.bashrc
```
####################### add line ####################
```bash
export OPENBLAS_CORETYPE=ARMV8
```

```bash
source ~/.bashrc
```
```bash
sudo pip3 install pillow
```
```bash
curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
```
```bash
mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl
```
```bash
sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl
```
```bash
sudo python3 -c "import torch; print(torch.cuda.is_available())"
```
### Installation of torchvision.

```bash
git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision
cd torchvision/
sudo python3 setup.py install
```
### Clone yolov5 Repositories and make it Compatible with Jetson Nano.

```bash
cd
git clone https://github.com/ultralytics/yolov5.git
cd yolov5/
```

``` bash
sudo pip3 install numpy==1.19.4
history
##################### comment torch,PyYAML and torchvision in requirement.txt##################################
sudo pip3 install --ignore-installed PyYAML>=5.3.1
sudo pip3 install -r requirements.txt
sudo python3 detect.py
sudo python3 detect.py --weights yolov5s.pt --source 0
```
## Queue Dataset Training
### We used Google Colab And Roboflow

train your model on colab and download the weights and past them into yolov5 folder
link of project

Insert gif or link to demo


## Running Queue Detection Model
source '0' for webcam

```bash
!python detect.py --weights best.pt --img 416 --conf 0.1 --source 0
```
## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


## Advantages

➢ Queue detection system will be of great help where efficiency is a priority.

➢ Queue detection system shows whether the group of people in viewfinder of camera
module are in Queue or not with good accuracy.

➢ When the Queue detection model finds a person out of the Queue it can be used with
a voice output like a speaker to announce particular person’s dress code like red shirt
and blue pants or yellow shirt and black pants or even a golden dress and tell them to
stay in line.

➢ Queue detection model works completely automated and no user input is required.

➢ It can work around the clock and therefore becomes more cost efficient.mal or very less workforce.
## Application

➢ Detects group of people and then checks whether they are in Queue or not in a given
image frame or viewfinder using a camera module.

➢ Can be used anywhere where Queue are formed like Ticket counters in railway station,
theatres, auditorium, aquarium, stadium etc.

➢ Can be used as a reference for other ai models based on Queue detection system.
## Future Scope


➢ As we know technology is marching towards automation, so this project is one of the
steps towards automation.

➢ Thus, for more accurate results it needs to be trained for more images, and for a greater
number of epochs.

➢ Queue detection will become a necessity in the future due to rise in population and
hence our model will be of great help to tackle the situation in an efficient way. As
population increase means longer Queue and hence increase in chance of jumping
Queue.
## Conclusion

➢ In this project our model is trying to detect group of people and then showing it on
viewfinder, live as to whether in Queue or not as we have specified in Roboflow.

➢ This model tries to solve the problem of Queue jumping by people and hence increases
efficiency as well as customer trust for people who are not trying to jump queue while
also making the chances of them being a regular higher.

➢ The model is efficient and highly accurate and hence reduces the workforce required
as no human need to keep an eye on Queue.
## Refrences

1]Roboflow :- https://roboflow.com/

2] Datasets or images used: https://www.gettyimages.ae/photos/queue-of-people?assettype=image&license=rf&alloweduse=availableforalluses&family=creative&phrase=queue%20of%20people&sort=best

3] Google images
## Articles

[1] https://pursuit.unimelb.edu.au/articles/now-we-know-why-we-stand-in-queues

[2] https://en.wikipedia.org/wiki/Queue_area
