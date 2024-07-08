document.addEventListener('DOMContentLoaded', function () {
  const predictBtn = document.getElementById('predict')
  const modelForm = document.getElementById('modelForm')
  console.log(modelForm)
  console.log(predictBtn)

  modelForm.addEventListener('submit', (e) => {
    e.preventDefault()
    console.log('hello qoyyum')

  })
  

  predictBtn.addEventListener('click', () => {
    console.log('hello qoyyum')
    // add function here
  })
})
