variable "profile" {
  description = "AWS Profile to use for API calls"
  type = string
  default = "default"
}

variable "role_name" {
  description = "AWS role name"
  type = string
  default = "mlem"
}

variable "user_name" {
  description = "AWS user name"
  type = string
  default = "mlem"
}

variable "region_name" {
  description = "AWS region name"
  type = string
  default = "us-east-1"
}

provider "aws" {
  region = var.region_name
  profile = var.profile
}

resource "aws_iam_user" "aws_user" {
  name = var.user_name
}

resource "aws_iam_access_key" "aws_user" {
  user = aws_iam_user.aws_user.name
}

resource "aws_iam_user_policy_attachment" "sagemaker_policy" {
  user = aws_iam_user.aws_user.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_user_policy_attachment" "ecr_policy" {
  user = aws_iam_user.aws_user.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess"
}

resource "aws_iam_role" "aws_role" {
  name = var.role_name
  description = "MLEM SageMaker Role"
  assume_role_policy = <<EOT
{"Statement": [{"Action": "sts:AssumeRole",
                                 "Effect": "Allow",
                                 "Principal": {
                                     "Service": "sagemaker.amazonaws.com"},
                                 "Sid": ""}],
                  "Version": "2012-10-17"}
  EOT
}

resource "aws_iam_role_policy_attachment" "sagemaker-access" {
  role = aws_iam_role.aws_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

output "secret_access_key" {
  value = aws_iam_access_key.aws_user.secret
  sensitive = true
}

output "access_key_id" {
  value = aws_iam_access_key.aws_user.id
  sensitive = true
}

output "region_name" {
  value = var.region_name
}

output "aws_user" {
  value = aws_iam_user.aws_user.name
}
