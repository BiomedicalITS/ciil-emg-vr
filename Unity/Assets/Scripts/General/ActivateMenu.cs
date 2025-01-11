using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ActivateMenu : MonoBehaviour
{
    public List<GameObject> gameObjectsToToggle; // List of game objects to toggle

    private void Start()
    {
        ToggleGameObjectsActivation();
    }

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.Return)) // Check if Enter key is pressed
        {
            ToggleGameObjectsActivation();
        }
    }

    void ToggleGameObjectsActivation()
    {
        foreach (GameObject gameObject in gameObjectsToToggle)
        {
            gameObject.SetActive(!gameObject.activeSelf); // Toggle the activation state
        }
    }
}
