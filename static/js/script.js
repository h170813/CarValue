document.addEventListener("DOMContentLoaded", function () {
    const carBrands = [
        'BMW', 'Audi', 'Ford', 'Toyota', 'Honda', 'Mercedes-Benz', 'Volkswagen',
        'Chevrolet', 'Nissan', 'Hyundai', 'Kia', 'Subaru', 'Mazda', 'Jeep', 'Lexus',
        'Dodge', 'Porsche', 'Ferrari', 'Lamborghini', 'Bentley', 'Rolls-Royce'
    ];

    const fuelTypes = [
        'Gasoline', 'Diesel', 'Hybrid', 'Electric'
    ];

    // Populate car brand suggestions
    const carBrandInput = document.getElementById('car_brands');
    carBrands.forEach(brand => {
        const option = document.createElement('option');
        option.value = brand;
        carBrandInput.appendChild(option);
    });

    // Populate fuel type dropdown
    const fuelTypeInput = document.getElementById('fuel_type');
    fuelTypes.forEach(fuel => {
        const option = document.createElement('option');
        option.value = fuel;
        option.textContent = fuel;
        fuelTypeInput.appendChild(option);
    });

    // Car brand validation to ensure only listed brands are chosen
    const carBrandTextInput = document.getElementById('car_brand');
    carBrandTextInput.addEventListener('blur', function () {
        const brand = carBrandTextInput.value;
        if (!carBrands.includes(brand)) {
            alert("Please select a valid car brand from the list.");
            carBrandTextInput.value = '';
        }
    });
});
