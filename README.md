# mlem-prototype
Project to share code ideas and concepts, track tasks and issues for upcoming MLEM tool

## Examples
[DVC Pipeline with mlem](examples/dvc-pipeline/README.md)

## Current state
Implemented mlem cli & api

### API
#### mlem.api.save
Saves object to fs in format of `<name>.mlem` file and `<name>` dir with artifacts. .mlem file contains all metadata needed to restore objects and some other fields, like requirements for models or columns and types for data frames
#### mlem.api.load
Loads object which was saved with `mlem.api.save`
### CLI
#### mlem apply
Usage: `mlem apply -m <method name> <model> <output> <inputs>`
Loads model and input data, applies `model.method` to it and saves result to output path in mlem format.

#### mlem deploy
##### mlem deploy `<model>` heroku
Deploys model to heroku. Needs HEROKU_API_KEY env (get it from heroku.com) and
and also this
```
REGISTRY_HEROKU_COM_PASSWORD=${HEROKU_API_KEY}
REGISTRY_HEROKU_COM_USERNAME=_
```
Deployment metadata is written to .mlem model file (subject to change in future)

##### mlem deploy `<model>` sagemaker --method predict
Deploys model to sagemaker. Need to set aws envs:
```
export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=
export AWS_DEFAULT_REGION=us-east-1
```

##### mlem deploy `<model>` status
Checks status of deployment. For now there is no conventions what it will return

##### mlem deploy `<model>` destroy
Undeploy deployed model. Deployment meta is removed from .mlem file

#### mlem apply-remote
Same as `mlem apply`, but actually sends data to deployed model

#### mlem pack `<model>` `<path>`
Generate model package to `<path>`


### API2
#### mlem env create `<name>` `<type>`
creates new target environment
type is one of `[sagemaker, heroku]`

#### mlem deploy2 `<model>` `<env_name>`
deploys model to chosen taget env
deploy metadata is saved to `<model>-<env_name>.deployed.yaml`

#### mlem destory2 `<deploy-name>`
destroy deploy described in some `<model>-<env_name>.deployed.yaml` file

#### mlem status2 `<deploy-name>`
get deployment status
