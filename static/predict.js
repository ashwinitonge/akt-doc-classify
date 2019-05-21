window.addEventListener('DOMContentLoaded', function(e){

	var predictForm = document.querySelector('.needs-validation');	
	if(predictForm) {
		predictForm.addEventListener('submit', function(e) {
		e.preventDefault();
		if(predictForm.checkValidity()) {
		predict();
		}
		predictForm.classList.add("was-validated");
		});
	}
});

function onSuccess(response){	
	console.log(response);
	
	var label = document.querySelector('#label');
	var conf = document.querySelector('#conf');
	var err = document.querySelector('#error');
	
	label.textContent = response.data.result['label']
	conf.textContent = response.data.result['confidence']
	err.classList.add("d-none");
	
	var result = document.querySelector('.js-result');	
	result.classList.remove("d-none");	
	
	resetButtonState();
}

function onError(error) {
	console.log(error);
	var err = document.querySelector('#error');
	err.classList.remove("d-none");
	resetButtonState();
}

function resetButtonState() {
	var predictButton = document.querySelector('.js-predict');
	var spinner = document.querySelector('.js-spinner');	
	var buttonText = document.querySelector('.js-predict-text');
	spinner.classList.add("d-none");
	buttonText.textContent = "Predict";
	predictButton.removeAttribute("disabled");
}

function predict() {
	var predictButton = document.querySelector('.js-predict');
	if (predictButton.hasAttribute("disabled")){
		return;
	}

	predictButton.setAttribute("disabled","disabled");
	var spinner = document.querySelector('.js-spinner');	
	var buttonText = document.querySelector('.js-predict-text');	

	spinner.classList.toggle("d-none");
	buttonText.textContent = "Predicting...";

	var words = document.querySelector('#words').value;
	
	axios.post('/predict', {
	"text": words
	})
	.then(function (response) {		
		onSuccess(response);
	})
	.catch(function (error) {
		onError(error);
	});
}