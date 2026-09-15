"""Example usage of the PoseTransfer API."""
from pose_transfer import PoseTransfer

if __name__ == "__main__":
    pt = PoseTransfer()
    result = pt.transfer_pose(
        target_images=["examples/target_person1.jpg", "examples/target_person2.jpg"],
        reference_image="examples/reference_pose.jpg",
        prompt="A person in the reference pose, wearing the target clothes",
        num_steps=30,
    )
    result.save("examples/result.jpg")
    print("Saved examples/result.jpg")
