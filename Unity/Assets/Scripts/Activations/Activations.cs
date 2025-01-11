using UnityEngine;

public class Activations : MonoBehaviour
{
    public void OnOff(GameObject obj)
    {
        if (obj.activeSelf)
        {
            obj.SetActive(false);
        }
        else
        {
            obj.SetActive(true);
        }
    }
}






