+++
title = "3. Train, tune, inference using TensorFlow"
date = 2020-02-14T12:20:47-05:00
weight = 2
+++

## Amazon SageMaker - TensorFlow / Keras Workshop using Image Classification

[Amazon SageMaker](https://aws.amazon.com/sagemaker/) provides every developer and data scientist the ability to build, train, and deploy machine learning models quickly. Amazon SageMaker is a fully-managed service that covers the entire machine learning workflow to label and prepare your data, choose an algorithm, train the model, tune and optimize it for deployment, make predictions, and take action. Your models get to production faster with much less effort and lower cost.

This workshop provides a set of lab exercises that demonstrate how machine learning developers and data scientists can leverage SageMaker to train, build, optimize, and deploy TensorFlow models using Keras. [TensorFlow, Keras and SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/tf.html) can be used for a wide variety of machine learning use cases. This workshop uses an Image Classification problem throughout the series of labs. When creating your notebook instance for running these labs, be sure to pick an instance type that has sufficient memory (try `ml.m5.large`, not `ml.t3.medium`). Also, if you want to train with the full dataset, we recommend choosing an instance type with GPU support (e.g., `ml.g4dn.xlarge`).

## Labs

### Using TensorFlow natively in a SageMaker Jupyter notebook
This introductory lab (1_tf_image_classification_birds.ipynb)
introduces the bird species image classification problem and acts as a **baseline of using Amazon SageMaker with TensorFlow**. The lab walks through preparing the data, and training an image classifier using transfer learning from a pretrained MobileNet model. It also evaluates the model. All this is done in the context of a SageMaker hosted Jupyter notebook. For someone familiar with TensorFlow and Keras, this is a straightforward first step of an end to end process performed directly from a SageMaker notebook.

### Leveraging SageMaker training and hosting with TensorFlow
Building on the introductory lab, this lab (2_tf_sm_image_classification_birds.ipynb) demonstrates the use of **SageMaker's TensorFlow container**. It shows that migration of existing TensorFlow models into the SageMaker platform is straightforward. SageMaker's managed training service is used to create training jobs on demand, reducing training cost by not paying for idle time. Similarly, the lab demonstrates realtime and batch deployment of TensorFlow models, providing an easy to use, scalable and cost effective approach to online and offline serving. Click [here](https://sagemaker.readthedocs.io/en/stable/using_tf.html) for details of using the TensorFlow container.



***
## Resources

The notebooks needed for the these labs can also be downloaded as a zip file: <a href="https://s3.amazonaws.com/ee-assets-prod-us-east-1/modules/05fa7598d4d44836a42fde79b26568b2/v2/tf_labs_v5.tar.gz">tf_labs_v5.tar.gz</a>
You can upload this file into a SageMaker Studio and extract it by running this command in the terminal `tar xzvf tf_labs_v5.tar.gz`