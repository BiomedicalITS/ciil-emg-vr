using UnityEngine;

[AddComponentMenu("Miscellaneous/README Info Note")]
public class CommentInformationNote : MonoBehaviour
{
    [TextArea(10, 1000)]
    public string Comment = "When generating a virtual bracelet, make sure to move all the cubes in the same PolygonContainer and to delete the other empty polygon container. See prefab for an example";
}
