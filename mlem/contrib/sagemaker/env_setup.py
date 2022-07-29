import os
import shutil
import subprocess

from mlem.ui import echo

MLEM_TF = "mlem_sagemaker.tf"

def _tf_command(tf_dir, command, *flags, **vars):
    vars = " ".join(f"-var='{k}={v}'" for k, v in vars.items())
    return " ".join(
        [
            # f"TF_DATA_DIR={tf_dir}",
            "terraform",
            f"-chdir={tf_dir}",
            command,
            *flags,
            vars,
        ]
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
    subprocess.check_output(_tf_command(work_dir, "init"), shell=True)

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
                profile=profile
            ),
            shell=True,
        )
    )

    if not plan and export_secret:
        if os.path.exists(export_secret):
            echo(
                f"Creds already present at {export_secret}, please backup and remove them"
            )
            return
        key_id = subprocess.check_output(
            _tf_command(work_dir, "output", "access_key_id"), shell=True
        ).decode("utf8")
        access_secret = subprocess.check_output(
            _tf_command(work_dir, "output", "secret_access_key"), shell=True
        ).decode("utf8")
        with open(export_secret, "w") as f:
            f.write(
                f"""export AWS_ACCESS_KEY_ID={key_id}
export AWS_SECRET_ACCESS_KEY={access_secret}
"""
            )