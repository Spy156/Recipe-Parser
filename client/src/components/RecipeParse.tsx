import React, { useState, useRef } from 'react';
import axios from 'axios';
import { Button, Input, Card, CardBody, CardHeader, Spinner, Image } from "@nextui-org/react";
import { Download, Youtube } from 'lucide-react';

// interface Ingredient {
//   name: string;
//   amount: string;
// }

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
      const response = await axios.post(`${import.meta.env.VITE_SERVER_ADDRESS}/predict`, {
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
      const response = await axios.post(`${import.meta.env.VITE_SERVER_ADDRESS}/analyze-image`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
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
        setError(`An error occurred while analyzing the image: ${error.response?.data?.error || error.message}`);
      } else {
        setError('An unexpected error occurred while analyzing the image. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownloadPDF = () => {
    console.log("Downloading PDF...");
  };

  return (
    <div className="container mx-auto p-4 max-w-4xl">
      <h1 className="text-4xl font-bold mb-8 text-center">Recipe Ingredient Parser</h1>
      <Card className="mb-8">
        <CardBody className="p-6">
          <div className="flex flex-col md:flex-row gap-4 mb-4">
            <Input
              label="Enter dish name"
              value={dishName}
              onChange={(e) => setDishName(e.target.value)}
              className="flex-grow"
            />
            <Input
              label="No. Of People"
              type="number"
              value={numPeople.toString()}
              onChange={(e) => setNumPeople(parseInt(e.target.value))}
              className="w-32"
            />
          </div>
          <div className="flex gap-4">
            <Button color="primary" onClick={handleParseRecipe}>
              Parse Recipe
            </Button>
            <Button color="secondary" onClick={() => fileInputRef.current?.click()}>
              Upload Image
            </Button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleImageUpload}
              accept="image/*"
              className="hidden"
            />
          </div>
        </CardBody>
      </Card>

      {selectedImage && (
        <Card className="mb-8">
          <CardBody>
            <Image
              src={URL.createObjectURL(selectedImage)}
              alt="Selected food"
              className="object-cover w-full h-[300px]"
            />
          </CardBody>
        </Card>
      )}

      {imagePredictions && (
        <Card className="mb-8">
          <CardBody>
            <h3 className="text-2xl font-bold mb-4">Image Predictions:</h3>
            <ul className="list-disc pl-6">
              {imagePredictions.map((pred, index) => (
                <li key={index} className="mb-2">
                  <b>{pred.class}:</b> {(pred.probability * 100).toFixed(2)}%
                </li>
              ))}
            </ul>
          </CardBody>
        </Card>
      )}

      {isLoading && (
        <Card className="mb-8">
          <CardBody className="flex items-center justify-center">
            <Spinner size="lg" />
          </CardBody>
        </Card>
      )}

      {error && (
        <Card className="mb-8">
          <CardBody className="bg-red-100 text-red-800">
            <p>{error}</p>
          </CardBody>
        </Card>
      )}

      {recipe && (
        <Card className="mb-8">
          <CardHeader className="pb-0 pt-6 px-6">
            <h2 className="text-3xl font-bold">{recipe.name}</h2>
          </CardHeader>
          <CardBody className="py-6 px-6">
            <p className="mb-4"><strong>Servings:</strong> {recipe.servings}</p>

            <h3 className="text-2xl font-bold mt-6 mb-4">Ingredients:</h3>
            <ul className="list-disc pl-6 mb-6">
              {recipe.ingredients.map((ingredient, index) => (
                <li key={index} className="mb-2">{ingredient}</li>
              ))}
            </ul>

            <h3 className="text-2xl font-bold mt-6 mb-4">Instructions:</h3>
            <ol className="list-decimal pl-6 mb-6">
              {recipe.instructions.map((step, index) => (
                <li key={index} className="mb-4">{step}</li>
              ))}
            </ol>

            <div className="flex gap-4 mt-8">
              <Button
                color="success"
                endContent={<Youtube size={20} />}
                onClick={() => window.open(`https://www.youtube.com/results?search_query=${encodeURIComponent(recipe.name)} recipe`, '_blank')}
              >
                Watch Videos
              </Button>
              <Button
                color="warning"
                endContent={<Download size={20} />}
                onClick={handleDownloadPDF}
              >
                Download PDF
              </Button>
            </div>
          </CardBody>
        </Card>
      )}
    </div>
  );
};

export default RecipeParser;