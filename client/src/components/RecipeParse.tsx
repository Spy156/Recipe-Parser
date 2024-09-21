import React, { useState, useRef } from 'react';
import axios from 'axios';

interface Ingredient {
  name: string;
  amount: string;
}

interface Recipe {
  name: string;
  ingredients: string[];
  instructions: string[];
  servings: number;
}

interface ImagePrediction {
  class: string;
  probability: number;
}

const RecipeParser: React.FC = () => {
  const [dishName, setDishName] = useState('');
  const [numPeople, setNumPeople] = useState(1);
  const [recipe, setRecipe] = useState<Recipe | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePredictions, setImagePredictions] = useState<ImagePrediction[] | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleParseRecipe = async () => {
    if (!dishName.trim()) {
      setError('Please enter a dish name or upload an image.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setRecipe(null);

    try {
      const response = await axios.post('http://localhost:5000/predict', {
        dish_name: dishName,
        num_people: numPeople
      });
      setRecipe(response.data);
    } catch (error) {
      console.error('Error parsing recipe:', error);
      setError('An error occurred while parsing the recipe. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      analyzeImage(file);
    }
  };

  const analyzeImage = async (file: File) => {
    setIsLoading(true);
    setError(null);
    setImagePredictions(null);

    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await axios.post('http://localhost:5000/analyze-image', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      console.log('Server response:', response.data);
      if (response.data.predictions) {
        setImagePredictions(response.data.predictions);
        if (response.data.predictions.length > 0) {
          setDishName(response.data.predictions[0].class);
        }
      }
      if (response.data.recipe) {
        setRecipe(response.data.recipe);
      }
    } catch (error) {
      console.error('Error analyzing image:', error);
      if (axios.isAxiosError(error)) {
        console.error('Axios error details:', error.response?.data);
        setError(`An error occurred while analyzing the image: ${error.response?.data?.error || error.message}`);
      } else {
        setError('An unexpected error occurred while analyzing the image. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-4">Recipe Ingredient Parser</h1>
      <div className="mb-4">
        <input
          type="text"
          value={dishName}
          onChange={(e) => setDishName(e.target.value)}
          placeholder="Enter dish name"
          className="border p-2 mr-2"
        />
        <input
          type="number"
          value={numPeople}
          onChange={(e) => setNumPeople(parseInt(e.target.value))}
          placeholder="Number of people"
          className="border p-2 mr-2"
        />
        <button onClick={handleParseRecipe} className="bg-blue-500 text-white p-2 rounded">
          Parse Recipe
        </button>
      </div>
      <div className="mb-4">
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleImageUpload}
          accept="image/*"
          className="hidden"
        />
        <button onClick={() => fileInputRef.current?.click()} className="bg-green-500 text-white p-2 rounded">
          Upload Image
        </button>
      </div>
      {selectedImage && (
        <img src={URL.createObjectURL(selectedImage)} alt="Selected food" className="max-w-xs mb-4" />
      )}
      {imagePredictions && (
        <div className="mb-4">
          <h3 className="text-xl font-bold">Image Predictions:</h3>
          <ul>
            {imagePredictions.map((pred, index) => (
              <li key={index}>
                {pred.class}: {(pred.probability * 100).toFixed(2)}%
              </li>
            ))}
          </ul>
        </div>
      )}
      {isLoading && <p>Loading...</p>}
      {error && <p className="text-red-500">{error}</p>}
      {recipe && (
        <div>
          <h2 className="text-2xl font-bold mb-2">{recipe.name}</h2>
          <p>Servings: {recipe.servings}</p>
          <h3 className="text-xl font-bold mt-4">Ingredients:</h3>
          <ul>
            {recipe.ingredients.map((ingredient, index) => (
              <li key={index}>{ingredient}</li>
            ))}
          </ul>
          <h3 className="text-xl font-bold mt-4">Instructions:</h3>
          <ol>
            {recipe.instructions.map((step, index) => (
              <li key={index}>{step}</li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
};

export default RecipeParser;