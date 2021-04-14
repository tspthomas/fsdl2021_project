$(document).ready(function() {
    setupImageClassification();
});

function showSpinner() {
    $("#spinner-loader").show();
}

function hideSpinner() {
    $("#spinner-loader").hide();
}

function setupImageClassification() {
    $('#image_classification_form').on('submit', (function(e) {
        e.preventDefault();

        var form_data = new FormData(this);
        
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
            },
            error: function(error)
            {
                hideSpinner();
                console.log(error);
            }
        });
    }));
}