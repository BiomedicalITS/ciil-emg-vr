using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using SFB;

public class SGTManager : MonoBehaviour
{
    public Slider _numRepsSlider;
    [SerializeField] private TextMeshProUGUI _numRepsSliderText;
    public Slider _timePerRepSlider;
    [SerializeField] private TextMeshProUGUI _timePerRepSliderText;
    public Slider _timeBetRepSlider;
    [SerializeField] private TextMeshProUGUI _timeBetRepSliderText;
    public TextMeshProUGUI _outputFolderText;
    public RawImage _inputCarrouselImage;
    [SerializeField] private TextMeshProUGUI _inputCarrouselText;

    [SerializeField] private GameObject _trainingPanel;
    [SerializeField] private Slider _timeSlider;
    [SerializeField] private TextMeshProUGUI _timeSliderText;
    [SerializeField] private RawImage _sgtCarrouselImage;
    [SerializeField] private TextMeshProUGUI _sgtCarrouselText;
    [SerializeField] private TcpClientManager _tcpClient;
    [SerializeField] private GeneralManager _generalManager;
    public EMGRawReader emgRawReader; // Public parameter to receive EMGRawReader

    private bool first_training = true;

    private int currentInputShown = 0;
    private string startstr = @"\Resources\";
    private string titlestartstr = @"Images\";

    private string names = "";

    private bool endTraining = false;

    private ExtensionFilter[] extensions = new[] {
        new ExtensionFilter("Images", "png", "jpg", "jpeg", "svg")
    };

    private string currentScene;

    void Start()
    {
        InitializeSlider(_numRepsSlider, _numRepsSliderText);
        InitializeSlider(_timePerRepSlider, _timePerRepSliderText);
        InitializeSlider(_timeBetRepSlider, _timeBetRepSliderText);
        InitializeSlider(_timeSlider, _timeSliderText);
        UpdateOutput();
        InitializeInputCarrousel();
        currentScene = SceneManager.GetActiveScene().name;
    }

    void Update()
    {

        if (Input.GetKey(KeyCode.LeftShift) && Input.GetKeyDown(KeyCode.Return))
        {
            StartTrainingClassifierClicked();
        }

        if (Input.GetKey(KeyCode.RightShift) && Input.GetKeyDown(KeyCode.Return))
        {
            NextSceneClicked();
        }
    }

    void SwitchScene(string sceneName)
    {
        if (currentScene != sceneName)
        {
            SceneManager.LoadScene(sceneName);

            currentScene = sceneName;
        }
    }

    void InitializeSlider(Slider _slider, TextMeshProUGUI _sliderText)
    {
        _slider.onValueChanged.AddListener((v) => {
            _sliderText.text = v.ToString("0");
        });
        _slider.value = PlayerPrefs.GetInt(_slider.ToString());
    }

    void SaveSlider(Slider _slider)
    {
        PlayerPrefs.SetInt(_slider.ToString(), (int)_slider.value);
    }

    void UpdateOutput()
    {
        _outputFolderText.text = PlayerPrefs.GetString("OutputFolder");
    }

    public void SaveAsDefaultClicked()
    {
        SaveSlider(_numRepsSlider);
        SaveSlider(_timePerRepSlider);
        SaveSlider(_timeBetRepSlider);
    }

    public void SelectOutputClicked()
    {
        var path = StandaloneFileBrowser.OpenFolderPanel("Select Output Folder", "", false);
        if (path.Length > 0)
        {
            PlayerPrefs.SetString("OutputFolder", path[0]);
            UpdateOutput();
        }
    }

    public void SelectInputClicked()
    {
        var paths = StandaloneFileBrowser.OpenFilePanel("Select Input Images", "", extensions, true);
        if (paths.Length > 0)
        {
            for (int i = 0; i < paths.Length; i++)
            {
                PlayerPrefs.SetString("InputImage" + i, paths[i]);
                string path = PlayerPrefs.GetString("InputImage" + i);
                int titlestart = path.IndexOf(titlestartstr, 0) + titlestartstr.Length;
                names += path.Substring(titlestart) + ",";
            }
            PlayerPrefs.SetString("InputsNames", names);
            PlayerPrefs.SetInt("InputCount", paths.Length);
            SetInputCarrousel(0);
        }
        else { PlayerPrefs.SetInt("InputCount", 0); }
    }

    public void ViewNextInputClicked()
    {
        if (PlayerPrefs.GetInt("InputCount") > 0)
        {
            if (currentInputShown == PlayerPrefs.GetInt("InputCount") - 1)
            {
                SetInputCarrousel(0);
            }
            else
            {
                SetInputCarrousel(currentInputShown + 1);
            }
        }
    }

    private void InitializeInputCarrousel()
    {
        if (PlayerPrefs.GetInt("InputCount") > 0)
        {
            SetInputCarrousel(0);
        }
    }

    public void SetInputCarrousel(int _index)
    {
        Texture2D pic = LoadPicture(_index);
        _inputCarrouselImage.texture = pic;
        string path = PlayerPrefs.GetString("InputImage" + _index);
        int titlestart = path.IndexOf(titlestartstr, 0) + titlestartstr.Length;
        _inputCarrouselText.text = _index.ToString() + "\n" + path.Substring(titlestart);
        currentInputShown = _index;
    }

    public void StartTrainingClassifierClicked()
    {
        _trainingPanel.SetActive(true);

        string prefix = "I";
        string message = prefix + " " + _numRepsSlider.value + " "
                        + _timePerRepSlider.value + " "
                        + _timeBetRepSlider.value + " "
                        + PlayerPrefs.GetString("InputsNames", "defaultInputs") + " "
                        + PlayerPrefs.GetString("OutputFolder", "defaultOutput");

        _tcpClient.SendMessageToServer(message);
        if (first_training)
            first_training = false;
        _generalManager.InitialTrainingConfig(message);
        StartCoroutine(Training());
    }
    /*
    public void StartTrainingCollectorClicked()
    {
        _trainingPanel.SetActive(true);

        string prefix = "O";
        string message = prefix + " " + _numRepsSlider.value + " "
                        + _timePerRepSlider.value + " "
                        + _timeBetRepSlider.value + " "
                        + PlayerPrefs.GetString("InputsNames", "defaultInputs") + " "
                        + PlayerPrefs.GetString("OutputFolder", "defaultOutput");

        _tcpClientCollector.SendMessageToServer(message);
        StartCoroutine(Training());
    }
    */
    public void StopTrainingClicked()
    {
        _trainingPanel.SetActive(false);
        endTraining = true;
    }

    public void NextSceneClicked()
    {
        emgRawReader.switchScene = true;
        SceneManager.LoadScene("DemoGab");
    }

    private IEnumerator Training()
    {
        for (int j = 0; j < _numRepsSlider.value; j++)
        {
            for (int i = 0; i < PlayerPrefs.GetInt("InputCount"); i++)
            {
                Texture2D pic = LoadPicture(i);
                Texture2D greypic = GetGreyPicture(pic);
                _sgtCarrouselImage.texture = greypic;
                string path = PlayerPrefs.GetString("InputImage" + i);
                int titlestart = path.IndexOf(titlestartstr, 0) + titlestartstr.Length;
                _sgtCarrouselText.text = "Rep " + (j + 1) + " of " + _numRepsSlider.value + "\nClass: " + path.Substring(titlestart);
                yield return StartCoroutine(RunFirstFunction());
                _sgtCarrouselImage.texture = pic;
                yield return StartCoroutine(RunNextFunction());
                if (endTraining)
                {
                    endTraining = false;
                    goto end;
                }
            }
            _tcpClient.SendMessageToServer("R");
        }
    end:
        StartCoroutine(Waiter());
    }

    IEnumerator Waiter()
    {
        yield return new WaitForSeconds(2);
        _tcpClient.SendMessageToServer("F");
        _trainingPanel.SetActive(false);
    }

    IEnumerator RunFirstFunction()
    {
        _timeSlider.maxValue = _timeBetRepSlider.value;
        float firstFunctionDuration = _timeBetRepSlider.value;
        float timer = 0.0f;

        while (timer < firstFunctionDuration)
        {
            int currentStep = Mathf.RoundToInt(timer);
            _timeSlider.value = currentStep;
            timer += Time.deltaTime;
            yield return null;
        }
    }

    IEnumerator RunNextFunction()
    {
        float nextFunctionDuration = _timePerRepSlider.value;
        _timeSlider.maxValue = _timePerRepSlider.value;
        float timer = 0.0f;
        _tcpClient.SendMessageToServer("S");

        while (timer < nextFunctionDuration)
        {
            int currentStep = Mathf.RoundToInt(timer);
            _timeSlider.value = currentStep;
            timer += Time.deltaTime;
            yield return null;
        }
        _tcpClient.SendMessageToServer("E");
    }

    private Texture2D LoadPicture(int _index)
    {
        string path = PlayerPrefs.GetString("InputImage" + _index);
        int start = path.IndexOf(startstr, 0) + startstr.Length;
        int end = path.IndexOf(".", start);
        string relpath = path.Substring(start, end - start);
        Texture2D pic = Resources.Load<Texture2D>(relpath);
        return pic;
    }

    private Texture2D GetGreyPicture(Texture2D _picture)
    {
        Texture2D greyPic = new Texture2D(_picture.width, _picture.height);

        for (int y = 0; y < _picture.height; y++)
        {
            for (int x = 0; x < _picture.width; x++)
            {
                Color pixelColor = _picture.GetPixel(x, y);
                float newR = (pixelColor.r + pixelColor.g + pixelColor.b) / 3f;
                float newG = newR;
                float newB = newR;
                Color newPixelColor = new Color(newR, newG, newB);
                greyPic.SetPixel(x, y, newPixelColor);
            }
        }
        greyPic.Apply();
        return greyPic;
    }
}
