import os
import shutil
import subprocess

from mlem.ui import echo

MLEM_TF = "mlem_sagemaker.tf"


def _tf_command(tf_dir, command, *flags, **args):
    args = " ".join(f"-var='{k}={v}'" for k, v in args.items())
    return " ".join(
        [
            "terraform",
            f"-chdir={tf_dir}",
            command,
            *flags,
            args,
        ]
    )


def _tf_get_var(tf_dir, varname):
    return (
        subprocess.check_output(
            _tf_command(tf_dir, "output", varname), shell=True  # nosec: B602
        )
        .decode("utf8")
        .strip()
        .strip('"')
    )


def sagemaker_terraform(
    user_name: str = "mlem",
    role_name: str = "mlem",
    region_name: str = "us-east-1",
    profile: str = "default",
    plan: bool = False,
    work_dir: str = ".",
    export_secret: str = None,
):
    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)

    shutil.copy(
        os.path.join(os.path.dirname(__file__), MLEM_TF),
        os.path.join(work_dir, MLEM_TF),
    )
    subprocess.check_output(
        _tf_command(work_dir, "init"),
        shell=True,  # nosec: B602
    )

    flags = ["-auto-approve"] if not plan else []

    echo(
        subprocess.check_output(
            _tf_command(
                work_dir,
                "plan" if plan else "apply",
                *flags,
                role_name=role_name,
                user_name=user_name,
                region_name=region_name,
                profile=profile,
            ),
            shell=True,  # nosec: B602
        )
    )

    if not plan and export_secret:
        if os.path.exists(export_secret):
            print(
                f"Creds already present at {export_secret}, please backup and remove them"
            )
            return
        key_id = _tf_get_var(work_dir, "access_key_id")
        access_secret = _tf_get_var(work_dir, "secret_access_key")
        region = _tf_get_var(work_dir, "region_name")
        profile = _tf_get_var(work_dir, "aws_user")
        print(profile, region)
        if export_secret.endswith(".csv"):
            secrets = f"""User Name,Access key ID,Secret access key
{profile},{key_id},{access_secret}"""
            print(
                f"Import new profile:\naws configure import --csv file://{export_secret}\naws configure set region {region} --profile {profile}"
            )
        else:
            secrets = f"""export AWS_ACCESS_KEY_ID={key_id}
export AWS_SECRET_ACCESS_KEY={access_secret}
export AWS_REGION={region}
"""
            print(f"Source envs:\nsource {export_secret}")
        with open(export_secret, "w", encoding="utf8") as f:
            f.write(secrets)
