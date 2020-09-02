Federated Learning (FL) is a privacy-preserving approach to train prdiction models without the requirement of  loading the data to the cloud. Since this novel field is application-oriented, this article aims to introduce the algorithms and technologies based on the so-far and promising applicatons. 

##  [Gboard](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html) 
Google is the pioneer of Federated Learning (FL) in both the research work and applications and Google Gboard is the most popular FL application for query suggestion. 

> "When Gboard shows a suggested query, your phone locally stores information about the current context and whether you clicked the suggestion."



So far, some algorithms and technologies have been proved efficient for Gboard. 

* Federated Averaging algorithm: This approach overcomes 



## Model Aggregation



## 

# tff Core

Aim:
* "To give researchers and practitioners explicit control over the specific patterns of distributed communication they will use in their systems"

* To provide "a flexible and extensible language for expressing distributed data flow algorithms"

* Abstraction Level: what data exists in the system and how it is transformed, but without dropping to the level of individual point-to-point network message exchanges.


Consist of:
* `distributed communication operators`: Federated values are opaque to programmers and 'are intended to be collectively transformed only by federated operators that abstractly represent various kinds of distributed communication protocols (such as aggregation)'.


Why need placement for "all-equal federated type, i.e. `T@G`? (e.g. tff.CLIENTS, tff.SEVER)
* Each of of devices would receive a separate set of instructions to execute locally, depending on the role it plays in the system (an end-user device, a centralized coordinator, an intermediate layer in a multi-tier architecture, etc.).  
    * Design the following without the programmer dealing with the raw data or identities of the individual participants
        * which subsets of devices execute what code
        * where different portions of the data might physically materialize.
    
* "The placement specifications are one of the mechanisms designed to" ",together with a static analysis of programs written in TFF, provide formal privacy guarantees for sensitive on-device data."
  
+ "This is especially important when dealing with, e.g., application data on mobile devices. Since the data is private and can be sensitive, we need the ability to statically verify that this data will never leave the device (and prove facts about how the data is being processed)."
  
* "placement is modeled as a property of data in TFF, rather than as a property of operations on data": 
    + some of the TFF operations span across locations, and run "in the network"
    
    
    
    

Implement Federated Training and Evaluation Algorithm <-> Relation to Federated Averaging in tff.learning?


Implement Analytics

Implement custom types of computations: 1. over distributed data; 2. over a set of client devices in the system; 3. with provided distributed communication operators

Broadcasting models and parameters to those devices