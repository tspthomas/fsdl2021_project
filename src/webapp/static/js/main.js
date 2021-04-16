$(document).ready(function() {
    setupImageClassification();
    setupEventListeners();
});

function setupImageClassification() {
    $('#image_classification_form').on('submit', (function(e) {
        e.preventDefault();

        var form_data = new FormData(this);
        
        clearClassificationResults();
        showSpinner();

        $.ajax({
            url: 'http://' + document.domain + ':' + location.port +'/api/v1/intelscenes/',
            type: 'POST',
            data : form_data,
            contentType: false,
            cache: false,
            processData: false,
            timeout: 60000,
            success: function(data)
            {
                hideSpinner();
                results = JSON.parse(data);
                console.log(results);
                buildClassificationResults(results);
                buildFeedbackForm(results);
                showClassificationResults();
                setupSendFeedback();
            },
            error: function(error)
            {
                hideSpinner();
                console.log(error);
            }
        });
    }));
}

function setupSendFeedback() {
    $('#feedback_form').on('submit', (function(e) {
        e.preventDefault();

        var form_data = $('form').serialize();

        clearClassificationResults();
        showSpinnerFeedback();
        
        $.ajax({
            url: 'http://' + document.domain + ':' + location.port +'/api/v1/feedback/',
            type: 'POST',
            data : form_data,
            cache: false,
            timeout: 60000,
            success: function(data)
            {
                hideSpinnerFeedback();
                results = JSON.parse(data);
                console.log(results);
                showSuccessAlert();
            },
            error: function(error)
            {
                hideSpinnerFeedback();
                console.log(error);
            }
        });
    }));
}

function setupEventListeners() {
    $('#image_to_classify').change(function(event) {  
        if (this.files && this.files[0]) {   
            var reader = new FileReader();
            var filename = $("#image_to_classify").val();
            filename = filename.substring(filename.lastIndexOf('\\')+1);
            reader.onload = function(e) {
                $('#img_to_upload').attr('src', e.target.result);
            }
            reader.readAsDataURL(this.files[0]);    
        }
    });
}

function buildClassificationResults(results) {
    var div_classification = document.createElement("div");
    div_classification.classList.add("p-4", "bg-light", "col-md-6");
    
    var prediction = results["prediction"];
    var h6 = document.createElement("h6");
    h6_content = "The uploaded image is a <span class='badge bg-primary'>"+ prediction +"</span>";
    h6.innerHTML = h6_content;

    div_classification.appendChild(h6);

    $("#classification-results").append(div_classification);
}

function createCorrectClassDiv() {
    var div = document.createElement("div");

    var h6 = document.createElement("h6");
    h6.classList.add("mt-3");
    h6_content = "Is the classification correct?";
    h6.innerHTML = h6_content;
    div.appendChild(h6);

    return div;
}

function createSelectClassDiv(results) {
    var div = document.createElement("div");
    div.style = "display: none";
    div.id = "feedback_no";

    var h6_hidden = document.createElement("h6");
    h6_hidden.classList.add("mt-3");
    h6_hidden_content = "What is the correct class?";
    h6_hidden.innerHTML = h6_hidden_content;
    div.appendChild(h6_hidden);
    
    var div_no = document.createElement("div");

    var classes = results["classes"];
    for (var i=0; i < classes.length; i++) {
        var cls = classes[i].toLowerCase();
        var checked = results["prediction"] == cls;
        var rb = createRadioButton("radio_"+cls, cls, "feedback_class", checked);
        div_no.append(rb);
    }

    div.appendChild(div_no);

    var image_input_hidden =  document.createElement("input");
    image_input_hidden.setAttribute("type", "hidden");
    image_input_hidden.setAttribute("id", "uploaded_image_id");
    image_input_hidden.setAttribute("name", "uploaded_image_id");
    image_input_hidden.value = results["uploaded_image_id"];
    div.appendChild(image_input_hidden);

    return div;
}

function buildFeedbackForm(results) {
    var form = document.createElement("form");
    form.id = "feedback_form";
    form.setAttribute("enctype", "multipart/form-data");

    var div = createCorrectClassDiv();
    var radio_yes = createRadioButton("radio_yes", "Yes", "feedback_correct", true);
    var radio_no = createRadioButton("radio_no", "No", "feedback_correct");
    
    div.appendChild(radio_yes);
    div.appendChild(radio_no);
    form.appendChild(div);

    var div_hidden = createSelectClassDiv(results);
    form.appendChild(div_hidden);

    var send_feedback_button = document.createElement("input");
    send_feedback_button.classList.add("btn", "btn-info", "mt-3");
    send_feedback_button.setAttribute("type", "submit");
    send_feedback_button.setAttribute("value", "Send feedback");

    form.appendChild(send_feedback_button);

    $("#classification-results").append(form);

    setupRadioListener(radio_yes);
    setupRadioListener(radio_no);
}

function setupRadioListener(radio) {
    radio.addEventListener("change", function(e){
        var value = this.children[0].value;

        if (value == "no") {
            showFeedbackClass();
        } else {
            hideFeedbackClass();
        }
    });
}

function createRadioButton(id, text, name, checked=false) {
    var div = document.createElement("div");
    div.classList.add("form-check");
    div.classList.add("form-check-inline");

    var input = document.createElement("input");
    input.classList.add("form-check-input");
    input.setAttribute("id", id);
    input.setAttribute("type", "radio");
    input.setAttribute("name", name);
    input.setAttribute("value", text.toLowerCase());
    input.checked = checked;

    var label = document.createElement("label");
    label.classList.add("form-check-label");
    label.setAttribute("for", id);
    label.innerHTML = text;

    div.appendChild(input);
    div.appendChild(label);

    return div;
}

function showSpinner() {
    $("#spinner-loader").show();
}

function hideSpinner() {
    $("#spinner-loader").hide();
}

function showSpinnerFeedback() {
    $("#spinner-loader-feedback").show();
}

function hideSpinnerFeedback() {
    $("#spinner-loader-feedback").hide();
}

function clearClassificationResults() {
    $("#classification-results").empty();
    hideSuccessAlert();
}

function showClassificationResults() {
    $("#classification-results").show();
}

function showFeedbackClass() {
    $("#feedback_no").show();
}

function hideFeedbackClass() {
    $("#feedback_no").hide();
}

function showSuccessAlert() {
    $("#success-feedback").show();
}

function hideSuccessAlert() {
    $("#success-feedback").hide();
}