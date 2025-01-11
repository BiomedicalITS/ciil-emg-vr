using System.Collections.Generic;
using UnityEngine;
//teest
public class ActiveAtStart : MonoBehaviour
{
    public List<GameObject> InactiveAtStart = new List<GameObject>();

    void Start()
    {
        foreach (GameObject go in InactiveAtStart)
        {
            go.SetActive(false);
        }
    }
}


