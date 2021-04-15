$(document).ready(function() {
    setupImageClassification();
    setupEventListeners();
});

function showSpinner() {
    $("#spinner-loader").show();
}

function hideSpinner() {
    $("#spinner-loader").hide();
}

function clearClassificationResults() {
    $("#classification-results").empty();
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

        var form_data = new FormData(this);
        
        alert(form_data);
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

function buildFeedbackForm(results) {
    var form = document.createElement("form");
    form.id = "feedback_form";

    var div = document.createElement("div");

    var h6 = document.createElement("h6");
    h6.classList.add("mt-3");
    h6_content = "Is the classification correct?";
    h6.innerHTML = h6_content;
    div.appendChild(h6);

    var radio_yes = createRadioButton("radio_yes", "Yes", "flexRadioDefault", true);
    var radio_no = createRadioButton("radio_no", "No", "flexRadioDefault");
    div.appendChild(radio_yes);
    div.appendChild(radio_no);

    form.appendChild(div);

    var div_hidden = document.createElement("div");
    div_hidden.style = "display: none";
    div_hidden.id = "feedback_no";

    var h6_hidden = document.createElement("h6");
    h6_hidden.classList.add("mt-3");
    h6_hidden_content = "What is the correct class?";
    h6_hidden.innerHTML = h6_hidden_content;
    div_hidden.appendChild(h6_hidden);
    
    var div_no = document.createElement("div");

    var classes = results["classes"];
    for (var i=0; i < classes.length; i++) {
        var cls = classes[i].toLowerCase();
        var checked = results["prediction"] == cls;
        var rb = createRadioButton("radio_"+cls, cls, "inlineRadioOptions", checked);
        div_no.append(rb);
    }

    div_hidden.appendChild(div_no);
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