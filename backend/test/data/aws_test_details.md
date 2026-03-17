# Amazon Web Services (AWS) — Comprehensive Overview

## Introduction

Amazon Web Services (AWS) is the world's most widely adopted cloud computing platform, offered by Amazon.com. Launched in 2006, AWS provides over 200 fully featured services from data centers distributed across 33 geographic regions worldwide. AWS serves millions of customers including startups, enterprises, and government agencies, enabling them to lower costs, become more agile, and innovate faster. The platform operates on a pay-as-you-go pricing model, eliminating the need for large upfront capital expenditures on physical infrastructure.

## Core Compute Services

### Amazon EC2 (Elastic Compute Cloud)

Amazon EC2 provides resizable virtual servers in the cloud. Users can launch instances with a variety of operating systems including Amazon Linux, Ubuntu, Windows Server, and Red Hat Enterprise Linux. EC2 offers multiple instance families optimized for different workloads: General Purpose (M-series), Compute Optimized (C-series), Memory Optimized (R-series and X-series), Storage Optimized (I-series and D-series), and Accelerated Computing (P-series and G-series with GPU support). Auto Scaling groups allow EC2 capacity to adjust automatically based on demand, ensuring applications maintain performance during traffic spikes while minimizing cost during idle periods.

### AWS Lambda

AWS Lambda is a serverless compute service that runs code in response to events without provisioning or managing servers. Lambda supports runtimes for Python, Node.js, Java, Go, Ruby, .NET, and custom runtimes via container images. Functions are triggered by events from services like API Gateway, S3, DynamoDB Streams, SNS, SQS, and CloudWatch Events. Lambda automatically scales from zero to thousands of concurrent executions and charges only for the compute time consumed, measured in millisecond increments. The maximum execution timeout is 15 minutes per invocation with up to 10 GB of memory allocation.

### Amazon ECS and EKS

Amazon Elastic Container Service (ECS) and Amazon Elastic Kubernetes Service (EKS) provide container orchestration. ECS is a fully managed service tightly integrated with the AWS ecosystem, while EKS runs upstream Kubernetes for teams that prefer Kubernetes-native tooling. Both support AWS Fargate, a serverless compute engine for containers that eliminates the need to manage underlying EC2 instances.

## Storage Services

### Amazon S3 (Simple Storage Service)

Amazon S3 is an object storage service offering industry-leading scalability, data availability, security, and performance. S3 stores data as objects within buckets, with each object supporting up to 5 terabytes. Storage classes include S3 Standard for frequently accessed data, S3 Intelligent-Tiering for automatic cost optimization, S3 Standard-IA and One Zone-IA for infrequent access, S3 Glacier Instant Retrieval, S3 Glacier Flexible Retrieval, and S3 Glacier Deep Archive for long-term archival at costs as low as $1 per terabyte per month. S3 provides 99.999999999% (eleven nines) durability by automatically replicating data across a minimum of three Availability Zones.

### Amazon EBS (Elastic Block Store)

Amazon EBS provides persistent block storage volumes for EC2 instances. Volume types include General Purpose SSD (gp3), Provisioned IOPS SSD (io2) delivering up to 256,000 IOPS, Throughput Optimized HDD (st1), and Cold HDD (sc1). EBS snapshots are stored incrementally in S3, and volumes can be encrypted using AWS Key Management Service (KMS).

### Amazon EFS (Elastic File System)

Amazon EFS is a fully managed, elastic NFS file system that can be mounted by multiple EC2 instances and Lambda functions simultaneously. It automatically grows and shrinks as files are added or removed, supporting petabyte-scale workloads with throughput up to 10 GB/s.

## Database Services

### Amazon RDS (Relational Database Service)

Amazon RDS manages relational databases including MySQL, PostgreSQL, MariaDB, Oracle, SQL Server, and Amazon Aurora. RDS automates backups, patching, scaling, and replication. Amazon Aurora is a MySQL and PostgreSQL-compatible database built for the cloud, delivering up to five times the throughput of standard MySQL and three times that of PostgreSQL, with storage that auto-scales up to 128 terabytes.

### Amazon DynamoDB

DynamoDB is a fully managed NoSQL key-value and document database delivering single-digit millisecond performance at any scale. It supports both on-demand and provisioned capacity modes, global tables for multi-region replication, and DynamoDB Streams for change data capture. DynamoDB Accelerator (DAX) provides an in-memory cache that reduces read latency to microseconds.

### Amazon ElastiCache

ElastiCache provides managed in-memory data stores compatible with Redis and Memcached. It is commonly used for caching, session management, real-time analytics, and leaderboards, reducing database load by serving frequently accessed data from memory.

## Networking

### Amazon VPC (Virtual Private Cloud)

Amazon VPC enables users to launch AWS resources in a logically isolated virtual network. VPC features include subnets (public and private), route tables, internet gateways, NAT gateways, VPC peering, transit gateways, and VPC endpoints for private connectivity to AWS services without traversing the public internet. Security groups act as virtual firewalls at the instance level, while Network ACLs provide stateless filtering at the subnet level.

### Amazon CloudFront

CloudFront is a global content delivery network (CDN) with over 450 edge locations. It accelerates delivery of static assets, dynamic content, APIs, and video streams. CloudFront integrates natively with AWS Shield for DDoS protection and AWS WAF for application-layer filtering.

### Elastic Load Balancing

AWS offers three load balancer types: Application Load Balancer (ALB) for HTTP/HTTPS traffic with path-based routing, Network Load Balancer (NLB) for ultra-low latency TCP/UDP traffic handling millions of requests per second, and Gateway Load Balancer (GWLB) for deploying third-party virtual appliances.

## Security and Identity

### AWS IAM (Identity and Access Management)

IAM controls access to AWS services and resources. It supports users, groups, roles, and policies written in JSON. IAM policies follow the principle of least privilege, granting only the permissions required. IAM roles enable cross-account access and federation with external identity providers via SAML 2.0 and OpenID Connect. Multi-factor authentication (MFA) adds a second layer of protection for privileged accounts.

### AWS KMS and Secrets Manager

AWS Key Management Service (KMS) creates and manages cryptographic keys used to encrypt data across AWS services. AWS Secrets Manager rotates, manages, and retrieves database credentials, API keys, and other secrets throughout their lifecycle, eliminating hard-coded credentials in application code.

## AI and Machine Learning

### Amazon SageMaker

SageMaker is a fully managed platform for building, training, and deploying machine learning models. It provides Jupyter notebook instances, built-in algorithms, distributed training, automatic model tuning (hyperparameter optimization), and real-time or batch inference endpoints. SageMaker Studio offers an integrated development environment for the entire ML workflow.

### Amazon Bedrock

Amazon Bedrock provides access to foundation models from AI21 Labs, Anthropic, Cohere, Meta, Stability AI, and Amazon Titan through a single API. It enables building generative AI applications with capabilities like text generation, summarization, image generation, and conversational agents, all without managing infrastructure.

## Monitoring and Management

### Amazon CloudWatch

CloudWatch collects and tracks metrics, logs, and events from AWS resources. It supports custom dashboards, alarms that trigger Auto Scaling or SNS notifications, and CloudWatch Logs Insights for querying log data using a purpose-built query language. Container Insights provides monitoring for ECS and EKS clusters.

### AWS CloudTrail

CloudTrail records API calls made across an AWS account, providing a complete audit trail for governance, compliance, and operational troubleshooting. Events are delivered to S3 buckets and can be analyzed with Athena or integrated into SIEM solutions.

## Pricing and Cost Management

AWS follows a pay-as-you-go model with no upfront commitments for most services. Reserved Instances and Savings Plans offer discounts of up to 72% for predictable workloads. The AWS Free Tier provides 12 months of limited free usage for new accounts, including 750 hours of EC2 t2.micro instances, 5 GB of S3 storage, and 25 GB of DynamoDB capacity per month. AWS Cost Explorer and AWS Budgets help organizations visualize spending patterns, set alerts, and optimize resource utilization.

## Conclusion

AWS provides a comprehensive and mature cloud platform that covers compute, storage, databases, networking, security, AI/ML, and operational management. Its global infrastructure, extensive service catalog, and flexible pricing make it suitable for workloads of any size — from a single Lambda function to enterprise-wide migrations involving thousands of servers and petabytes of data.
