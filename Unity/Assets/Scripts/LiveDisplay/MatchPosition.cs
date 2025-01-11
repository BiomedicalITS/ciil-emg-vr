using UnityEngine;

public class MatchTransform : MonoBehaviour
{
    public GameObject objectB; // Reference to the GameObject (forearm) whose position and rotation you want to match
    public Vector3 positionOffset = Vector3.zero; // Offset to be applied to the position

    void Update()
    {
        // Check if the reference to objectB is not null
        if (objectB != null)
        {
            // Get the current rotation of ObjectB
            Quaternion newRotation = objectB.transform.rotation;

            // Add a 45-degree rotation to the x-axis
            newRotation *= Quaternion.Euler(-45f, 0f, 0f);

            // Match the position and modified rotation of this GameObject to objectB with the offset
            transform.position = objectB.transform.position + positionOffset;
            transform.rotation = newRotation;
        }
        else
        {
            // Log an error message if objectB is not assigned
            Debug.LogError("ObjectB reference is null. Please assign a GameObject to the ObjectB variable in the inspector.");
        }
    }
}
