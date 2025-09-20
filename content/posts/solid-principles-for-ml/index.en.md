---
title: SOLID Principles Explained with ML Code
date: 2025-02-14T17:24:18+09:00
draft: false
author: pizzathief
description:
summary: OOP made less scary (and more useful) for data practitioners
keywords:
  - solid
  - 객체 지향 프로그래밍
  - oop
tags:
  - AI-ML
categories:
  - data
slug: solid-principles-for-ml
ShowReadingTime: true
ShowShareButtons: true
ShowPostNavLinks: false
ShowBreadCrumbs: false
ShowCodeCopyButtons: true
ShowWordCount: false
ShowRssButtonInSectionTermList: true
disableScrollToTop: false
hidemeta: false
hideSummary: false
showtoc: true
tocopen: false
UseHugoToc: true
disableShare: false
searchHidden: false
robotsNoIndex: false
comments: true
weight: 10
math: true
---


## You Love Jupyter, But It Won’t Love You Back in Production


{{< figure src="/posts/solid-principles-for-ml/stop-it-patrick.png" width="300" align="center">}}

Jupyter Notebook is an incredibly handy tool for anyone doing data analysis or building ML models. Unlike regular software development, a huge chunk of the work here revolves around experiments. So when we write code, we’re often not that concerned with the overall architecture or design—what we really want is to just run it quickly and check: _what does this distribution look like?_ or _how’s the accuracy shaping up?_ That’s pretty much the mindset we all share. And that’s exactly why Jupyter, with its cell-based execution and instant feedback, feels like such a perfect fit—it almost doesn’t need explaining. Of course, no one reading this is probably “scared” of functions, but the truth is we often end up writing everything in one long, unstructured flow. Not because "functions scare us", but simply because… well, it’s just more convenient that way.

But once you get past quick ad-hoc analysis or early-stage prototyping and actually need to ship something to production, Jupyter Notebooks can quickly turn into a real headache. A single notebook stuffed with everything—data preprocessing, model training, inference, even data loading—ends up as this big monolithic blob of code that’s almost impossible to deal with. And honestly, it’s always a bit funny to see developers from other teams trying (and failing) to hide the look of shock when they stumble across one of these notebooks.

The problems with this kind of setup are too many to list, but the big ones are:
- **Debugging and maintenance are a nightmare** (Have you ever opened a notebook with 89 cells? Be real—even if it’s your _own_ code from a few months back, there’s no way you’ll remember what’s where.)
- **Version control and collaboration are painful** (Comparing changes is clunky, and since outputs are saved in the file, the whole thing looks different even when the logic hasn’t really changed.)
- **Reuse and extension are nearly impossible** (There’s no clear separation of functionality, logic gets copy-pasted everywhere, and if you want to tweak one specific feature, you’re basically out of luck.)


So even if you do all your experimentation in Jupyter, once it’s time to move toward production you really need a reasonable level of modularization and proper scripts. A useful reference at that point is the set of design guidelines from object-oriented programming (OOP) known as the **SOLID principles**.

There are five of them, and the name “SOLID” is just an acronym made from their initials. These principles have been around forever, so you’ll find no shortage of resources if you search for them—but the explanations can feel a bit abstract, and the examples don’t always click. That’s why today I want to walk through them using some simple, more relatable examples that made sense to me.


```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# load data
penguins = sns.load_dataset("penguins")

# preprocess
penguins.dropna(inplace=True)
le = LabelEncoder()
penguins['species'] = le.fit_transform(penguins['species'])
penguins['sex'] = le.fit_transform(penguins['sex'])
penguins['island'] = le.fit_transform(penguins['island'])
X = penguins.drop(columns=['species'])
y = penguins['species']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model train and evaluation
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", hue="species")
plt.title("Penguin Species based on Bill Length and Depth")
plt.show()
```

  
The code above loads the penguin dataset and does a simple species classification—basically the ABCs of machine learning, nothing you need to study too closely. It’s also a textbook example of what I called “Jupyter-style” code earlier. Now, let’s try refactoring it with the SOLID principles in mind.

- **Quick disclaimer before we dive in 📍**
    - The refactored versions are _not_ meant to say “this is the only right way to write code.” They’re not super polished, and in a real project I’d probably take a different approach. Think of them more as **examples to help explain SOLID principles in a language that feels familiar to folks doing data analysis and modeling.**
    - Also, it’s not that every single SOLID rule must always be followed in an ML project. Rather, see them as a general direction you can aim for when writing production-ready code.

<br>

## Single Responsibility Principle

> **A class should have only one responsibility.**

“Responsibility” might sound vague, but you can think of it simply as a _function_ or _role_. In other words, each class should focus on doing one thing well. That also means whenever you change the class, it should only be because there’s a change related to that one specific responsibility—one reason, one change.


```python
class DataLoader:
	def load_data():
		return sns.load_dataset("penguins")

class DataPreprocessor:
	def preprocess_data(df):
		df.dropna(inplace=True)
		le = LabelEncoder()
		df['species'] = le.fit_transform(df['species'])
		df['sex'] = le.fit_transform(df['sex'])
		df['island'] = le.fit_transform(df['island'])
		X = df.drop(columns=['species'])
		y = df['species']
		return X, y


class ModelTrainer:
	def __init__(self, model):
		self.model = model
	def train(self, X_train, y_train):
		self.model.fit(X_train, y_train)
	def predict(self, X_test):
		return self.model.predict(X_test)
	def evaluate(self, y_test, y_pred):
		return accuracy_score(y_test, y_pred)

class DataVisualizer:
	def visualize(df):
		plt.figure(figsize=(8, 6))
		sns.scatterplot(data=df, x="bill_length_mm", y="bill_depth_mm", hue="species")
		plt.title("Penguin Species based on Bill Length and Depth")
		plt.show()


if __name__ == "__main__":
	data = DataLoader.load_data()
	X, y = DataPreprocessor.preprocess_data(data)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	model = RandomForestClassifier(n_estimators=100, random_state=42)
	trainer = ModelTrainer(model)
	trainer.train(X_train, y_train)
	y_pred = trainer.predict(X_test)
	accuracy = trainer.evaluate(y_test, y_pred)
	print(f'Accuracy: {accuracy:.2f}')
	DataVisualizer.visualize(data)
```

In our original code, we didn’t even have any classes to begin with. So the first step was to break things down into separate classes, each responsible for exactly one piece of functionality:

- **DataLoader** – loads the dataset
- **DataPreprocessor** – handles preprocessing
- **ModelTrainer** – takes care of training, prediction, and evaluation
- **DataVisualizer** – deals with visualization

Now each class has **one clear role**. Which means if you ever want to tweak how the plots look, you only need to touch DataVisualizer—and nothing else.

<br>

## Open-Closed Principle

> **Software components (classes, modules, functions, etc.) should be open for extension, but closed for modification.**

At first glance, you might think, _“Wait—open in one place and closed in another? Isn’t that a contradiction?”_ But what it really means is that your code should be designed so you **don’t have to rewrite existing logic** when adding new functionality. Instead, you should be able to _extend_ it.
The key to achieving this is through **abstraction and inheritance**—define an abstract interface, and then build new features by implementing or subclassing it, rather than tinkering with the original code.


{{< figure src="/posts/solid-principles-for-ml/oop.jpg" width="400" caption="[image source](https://www.cs.sjsu.edu/~pearce/modules/lectures/ood/principles/ocp.htm)" align="center">}}


Let’s pause for a second to cover a core idea in object-oriented programming (OOP). Imagine a game with different characters—say, a mage and a warrior. A mage attacks with a staff, while a warrior swings a sword. The details differ, but the common point is that **they both need to be able to attack**.
So what we do is define an abstract **Character** interface with the basic abilities like move, attack, and defend. Then the Mage and Warrior classes **inherit** from this interface and implement their own versions of those actions. This is what we call **polymorphism**—and the beauty of it is that you can treat all these classes through the same interface, while still being able to extend them flexibly.
For example, if you need a new dragon character, you don’t have to tear down the Character interface or start over. You just inherit from it and implement a new “fire-breath” attack. That’s the essence of the Open-Closed Principle: leverage polymorphism so you can extend your system without rewriting the foundations.


```python
from abc import ABC, abstractmethod

class BaseModel(ABC):
	@abstractmethod
	def train(self, X_train, y_train):
		pass
	@abstractmethod
	def predict(self, X_test):
		pass
	@abstractmethod
	def evaluate(self, y_test, y_pred):
		pass

class RandomForestTrainer(BaseModel):
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, y_test, y_pred):
        return accuracy_score(y_test, y_pred)

class GradientBoostingTrainer(BaseModel):
    def __init__(self, learning_rate=0.1, n_estimators=100, random_state=42):
        self.model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, random_state=random_state)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, y_test, y_pred):
        return accuracy_score(y_test, y_pred)
```

Now, coming back to our example… First off, **ABC** is a Python library you use to define abstract classes and methods. An abstract class isn’t meant to be instantiated directly—instead, it serves as an interface that tells any subclasses, _“Here are the methods and properties you need to implement.”_ It’s basically the same idea as the **Character** class from the earlier example.  
The `@abstractmethod` decorator enforces this rule: if a class inherits from the abstract base, it **must** implement that method.
In our case, we created a BaseModel abstract class. Any actual model we want to use will inherit from this class. The contract is that subclasses have to implement train, predict, and evaluate as core functionalities. That way, if at some point you decide to swap out RandomForest for, say, a Gradient Boosted Tree (GBT), you don’t rewrite or modify BaseModel—you just inherit from it and provide the new implementation. That’s how you achieve _extension without modification_.

<br>

## Liskov Substitution Principle

> **Subclasses should be substitutable for their base classes.**

I’ll admit, this one didn’t click for me at first. When I first read it, I thought: _“Why would a subclass need to replace its parent? Didn’t we create subclasses to override and specialize behavior?”_ But the principle isn’t telling you to actually **replace** things—it’s saying you should **design your inheritance so that replacement is always possible without breaking anything**.
If you trace it back to its original definition, it goes something like: _“If a property holds true for objects of type T, it must also hold true for objects of type S, where S is a subtype of T.”_ In other words, everything that works for the base class should also work for its subclass. So the key takeaway is: **if swapping out a base class with one of its subclasses causes errors or weird behavior, then your inheritance hierarchy isn’t designed correctly.**


{{< figure src="/posts/solid-principles-for-ml/lsp.jpg" width="420" caption="[image source](https://scientificprogrammer.net/2019/11/04/liskov-substitution-principle-in-c/)" align="center">}}

If you build a toy duck that looks and quacks like a real duck, there’s a catch: the real duck waddles on its own when you set it down, but the toy duck won’t move without batteries. Drop the toy duck into the same situation as the real duck (i.e., code written against the base class), and it misbehaves—that’s a violation of LSP. What do you do? If you can make it move without batteries (i.e., rewrite the subclass to meet the base class’s guarantees), great. Usually you can’t, so the fix is to **reclassify**: make the toy duck a subtype of **Toy**, not of **Duck**, so it no longer promises behaviors it can’t deliver.
Back to our running example—this time, let’s tweak the **data preprocessing** side rather than the model.


```python
class DataPreprocessor(ABC):
    def __init__(self, df: pd.DataFrame):
        self.df = df
    @abstractmethod
    def preprocess(self) -> pd.DataFrame:
        """전처리 실행"""
        pass

class NumericPreprocessor(DataPreprocessor):
    def preprocess(self) -> pd.DataFrame:
        scaler = StandardScaler()
        numeric_columns = self.df.select_dtypes(include=["number"]).columns
        self.df[numeric_columns] = scaler.fit_transform(self.df[numeric_columns])
        return self.df 

class CategoryPreprocessor(DataPreprocessor):
    def preprocess(self) -> pd.DataFrame:
        encoder = OneHotEncoder(sparse=False)
        categorical_columns = self.df.select_dtypes(include=["object"]).columns
        if categorical_columns.empty:
            return self.df
        transformed = encoder.fit_transform(self.df[categorical_columns])
        column_names = encoder.get_feature_names_out(categorical_columns)
        encoded_df = pd.DataFrame(transformed, columns=column_names, index=self.df.index)
        self.df = self.df.drop(columns=categorical_columns).reset_index(drop=True)
        encoded_df = encoded_df.reset_index(drop=True)
        return pd.concat([self.df, encoded_df], axis=1)

class TrainTestSplitter(DataPreprocessor):
    def preprocess(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=42)
        return train_df, test_df 
```

From the original class with hard-coded column names, I introduced a top-level DataPreprocessor interface with SRP and OCP in mind, then subclassed it to handle preprocessing for **numeric** and **categorical** columns separately. The final step of preprocessing is splitting the data into train and test. Now, if we make the class that performs this split inherit from DataPreprocessor, we’d be violating LSP. Why? Because instead of returning a **preprocessed pandas DataFrame** like DataPreprocessor promises, it returns a **tuple of split datasets**. It might still “run,” but the subclass can’t substitute the base class—so the inheritance design is off.


<br>

## Interface Segregation Principle

> **Clients should not be forced to depend on interfaces they don’t use.**

What this means in practice is that interfaces should be kept as small and specific as possible, exposing only the methods that are truly needed. In other words, don’t cram a bunch of “just in case” methods into a single interface—design it so that each client only deals with the functionality it actually uses.



{{< figure src="/posts/solid-principles-for-ml/isp.png" width="450" caption="A base class interface defines two features, but from the subclass’s perspective, one of them isn’t actually needed. ([image source](https://www.linkedin.com/pulse/interface-segregation-principle-typescript-dhananjay-kumar/))" align="center">}}


This principle is pretty straightforward, so let’s skip the theory and jump into an example. Back in the OCP section, we created a BaseModel class that bundled together training, prediction, and evaluation. But if you think about it, not every case really needs all three.
For instance, you might want to load a pre-trained model and only run predictions. Or maybe you’re using it as a feature extractor—train it, then just grab the learned vectors without doing any predictions. And in an online learning scenario, the evaluation step might not even be relevant.

With that in mind, let’s break those three capabilities into separate interfaces.

```python
class Trainable(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass

class Predictable(ABC):
    @abstractmethod
    def predict(self, X_test):
        pass

class Evaluable(ABC):
    @abstractmethod
    def evaluate(self, y_test, y_pred):
        pass
```


Once the three interfaces are split out, you can mix and match—implementing only the pieces you actually need for each use case.

```python
class RandomForestTrainer(Trainable, Predictable, Evaluable):
	def __init__(self, n_estimators=100, random_state=42):
		self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
		
	def train(self, X_train, y_train):
		self.model.fit(X_train, y_train)
	
	def predict(self, X_test):
		return self.model.predict(X_test)
	
	def evaluate(self, y_test, y_pred):
		return accuracy_score(y_test, y_pred)

class PretrainedModel(Predictable):
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
    
    def predict(self, X_test):
		return self.model.predict(X_test)

class FeatureExtractor(Trainable):
    def __init__(self, n_components=1):
        self.model = PCA(n_components=n_components)
    
    def train(self, X_train):
        self.model.fit(X_train)
    
    def extract_features(self, X):
        return self.model.transform(X) 
```



<br>

## Dependency Inversion Principle

> **High-level modules should not depend on low-level modules. Both should depend on abstractions.**

{{< figure src="/posts/solid-principles-for-ml/dip.png" width="400" caption="[image source](https://levelup.gitconnected.com/solid-programming-for-arduino-the-dependency-inversion-principle-4ce3bdb787d1)" align="center">}}


So what exactly are “high-level” and “low-level” modules? A high-level module (A) is the class that runs the core business logic, while a low-level module (B) handles more detailed, concrete tasks that A calls and uses.
The Dependency Inversion Principle says A shouldn’t depend directly on B. Instead, there should be an abstract interface (C) sitting in between. The term _inversion_ comes from the idea that the high-level module doesn’t go looking for a specific low-level implementation—instead, the low-level piece gets **injected from the outside** through that abstraction.

```python
class MargheritaPizza:
    def prepare(self):
        print("🍅🌿🧀")

    def bake(self):
        print("🔥")

    def serve(self):
        print("🍕")

class PizzaSell:
    def __init__(self):
        self.pizza = MargheritaPizza() 

    def order_pizza(self):
        self.pizza.prepare()
        self.pizza.bake()
        self.pizza.serve()
```

In this example, each individual pizza is a low-level module, while the pizza sales logic is the high-level module. If the high-level module depends directly on specific pizza classes, then every time you add a new pizza to the menu, you’re forced to modify that class(It also breaks the Open-Closed Principle).

```python
class Pizza(ABC):
    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def bake(self):
        pass

    @abstractmethod
    def serve(self):
        pass

class HawaiianPizza(Pizza):
    def prepare(self):
        print("🍍🍖🧀")

    def bake(self):
        print("🔥")

    def serve(self):
        print("🍕")

class PizzaSell:
    def __init__(self, pizza: Pizza):  
        self.pizza = pizza 

    def order_pizza(self):
        self.pizza.prepare()
        self.pizza.bake()
        self.pizza.serve()

margherita_sell = PizzaSell(MargheritaPizza())
pepperoni_sell = PizzaSell(HawaiianPizza())
```

In the code above, the high-level module doesn’t depend on individual pizza types; it depends on an abstract Pizza interface. That gives you extensibility and easier maintenance—you can add a new pizza without touching the high-level logic.

Back to our original example and the **data preprocessing** part: so far we only introduced an abstract DataPreprocessor interface. We’ll keep the existing DataPreprocessor, NumericPreprocessor, and CategoricalPreprocessor as they are, and then add a **high-level module** on top that orchestrates them, like this.


```python
class PreprocessingPipeline:
    def __init__(self, preprocessors: list[DataPreprocessor]):
        self.preprocessors = preprocessors

    def run(self) -> pd.DataFrame:
        processed_df = self.preprocessors[0].df  
        for preprocessor in self.preprocessors:
            processed_df = preprocessor.preprocess()  
        return processed_df


pipeline = PreprocessingPipeline([NumericPreprocessor(df), CategoryPreprocessor(df)])
processed_df = pipeline.run()
```

The `pipeline` acts as a high-level module that takes in a series of low-level modules, all of type DataPreprocessor. This way, even if you later introduce a new preprocessing class—beyond the numeric scaler or categorical encoder you already have—it can plug right in without touching the existing code. In other words, the system stays flexible and extensible.

<br>

## References
- [객체 지향 설계의 5가지 원칙 - S.O.L.I.D](https://inpa.tistory.com/entry/OOP-💠-객체-지향-설계의-5가지-원칙-SOLID#)