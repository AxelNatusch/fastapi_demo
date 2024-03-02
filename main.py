from enum import Enum
from typing import Annotated, Any, List, Set

from fastapi import Body, Cookie, FastAPI, Header, Path, Query, Response
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, EmailStr, Field

# Query parameters (e.g. /items/?skip=0&limit=10)
# Path parameters (e.g. /items/5)
# Request body (e.g. the JSON body of a request to create an item)

# All are declared in the function parameters
# The function parameters will be recognized by FastAPI as follows:
# -> If the parameter is also declared in the path, it will be used as a path parameter.
# -> If the parameter is of a singular type (like int, float, str, bool, etc) it will be interpreted as a query parameter.
# -> If the parameter is declared to be of the type of a Pydantic model, it will be interpreted as a request body.

# Annotations (Annotated)
# -> Annotated can be used to declare additional validation and metadata for the parameters.
# -> Query class is used to declare the validation and metadata for query parameters.
# -> Path class is used to declare the validation and metadata for path parameters.
# -> Body class is used to declare the validation and metadata for request bodies.

# Field class (from Pydantic)
# -> Field can be used to declare additional validation and metadata for the values of a Pydantic model.


# Other
# - Header() -> to declare a header parameter. FastAPI will convert _ to - in the header name.
# -> multiple headers with the same name are allowed and FastAPI will recognize them as a list of values.
# - Cookie()
# - File()
# - Form()


# Return types (response model)
# -> FastAPI will use the return type of the function to determine the response model.
# -> If the function returns a Pydantic model, FastAPI will use it as the response model.
# -> If the function returns a dict, FastAPI will use it as the response model.

# You can also use the response_model parameter to declare the response model directly in the path operation decorator.
# -> response_model can be used to declare the response model directly in the path operation decorator.


app = FastAPI()


class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


class Image(BaseModel):
    url: str
    name: str


class Item(BaseModel):
    name: str
    description: str | None = Field(
        default=None, title="The description of the item", max_length=300
    )
    price: float = Field(
        gt=0, description="The price must be greater than zero", examples=[100, 200]
    )
    tax: float | None = None
    tags: Set[str] = set()
    image: Image | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Foo",
                    "description": "a very nice item",
                    "price": 100,
                    "tax": 19,
                },
            ]
        }
    }


class User(BaseModel):
    username: str
    full_name: str | None = None


class UserIn(BaseModel):
    username: str
    password: str
    email: EmailStr
    full_name: str | None = None


class UserOut(BaseModel):
    username: str
    email: EmailStr
    full_name: str | None = None


class BaseUser(BaseModel):
    username: str
    email: EmailStr
    full_name: str | None = None


class BaseUserIn(BaseUser):
    password: str


fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]


@app.get("/portal")
async def get_portal(teleport: Annotated[bool, Query()] = False) -> Response:
    if teleport:
        return RedirectResponse(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    return JSONResponse(content={"message": "Welcome to the portal"})


# response_model=None -> to skip the creation of the fastapi response model, so we can return either a Response or a dict
# -> FastAPI will usually create a response model for the return type of the function, but if this is not possible, you
# can use response_model=None to skip the creation of the response model.
@app.get("/portal_", response_model=None)
async def get_portal_(teleport: Annotated[bool, Query()] = False) -> Response | dict:
    if teleport:
        return RedirectResponse(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    return {"message": "Welcome to the portal"}


# wrong way as we would also send the password in the response
@app.post("/user")
async def create_user_wrong(user: UserIn) -> UserIn:
    return user


# better way but still not the best practice
# response_model overrides the return type of the function so FastAPI will use it as the response model.
# -> FastAPI will use the return type of the function to determine the response model.
@app.post("/user/", response_model=UserOut)
async def create_user_correct(user: UserIn) -> Any:
    return user


# Best practice to use inheritance if i want to use the same model for input and output but with different fields
@app.post("/user_correct/")
async def create_user(user: BaseUserIn) -> BaseUser:
    return user


# use pydantics include/ exclude functionality to exclude the password from the response
@app.post(
    "/user_also_cool/", response_model=UserIn, response_model_exclude={"password"}
)
async def create_user_also_cool(user: UserIn) -> Any:
    return user


@app.post("/items/")
async def create_itemmm(item: Item) -> Item:
    return item


@app.post("/itemss/", response_model=Item)
async def create_itemm(item: Item):
    return item


@app.put("/items/{item_id}")
async def update_itemm(
    item_id: int,
    item: Item,
    user: User,
    importance: Annotated[int, Body()],
    user_agent: Annotated[str | None, Header()] = None,
    ads_id: Annotated[str | None, Cookie()] = None,
    x_token: Annotated[List[str] | None, Header()] = None,
):
    """
    The function parameters will be recognized by FastAPI as follows:
    -> item_id will be recognized as a path parameter.
    -> item will be recognized as a request body.
    -> user will be recognized as a request body.
    -> importance will be recognized as a request body instead of a query parameter because of the Body() class.
    -> user_agent will be recognized as a header parameter.
    -> ads_id will be recognized as a cookie parameter.

    expected request body structure:
    {
        "item": {
            "name": "string",
            "description": "string",
            "price": 0,
            "tax": 0
        },
        "user": {
            "username": "string",
            "full_name": "string"
        },
        "importance": 5
    }
    """
    if x_token:
        for token in x_token:
            print(f"Token: {token}")

    result = {
        "item_id": item_id,
        "item": item,
        "user": user,
        "importance": importance,
        "user_agent": user_agent,
        "ads_id": ads_id,
    }

    return result


@app.put("/itemsss/{item_id}")
async def update_item(
    item_id: Annotated[int, Path(description="The ID of the item to get", gt=0)],
    item: Annotated[Item, Body(embed=True)],
):
    """
    The function parameters will be recognized by FastAPI as follows:
    -> item_id will be recognized as a path parameter.
    -> item will be recognized as a request body.

    expected request body structure (embed=True):
    {
        "item": {
            "name": "string",
            "description": "string",
            "price": 0,
            "tax": 0
        }
    }

    instead of:
    {
        "name": "string",
        "description": "string",
        "price": 0,
        "tax": 0
    }

    """
    results = {"item_id": item_id, "item": item}
    return results


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/items/")
async def create_item(item: Item):
    item_dict = item.model_dump()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict


# type hints result in automatic validation and transformation (using pydantic in the background)
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}


@app.get("/items/{item_id}")
async def read_item_q(item_id: str, q: str | None = None, short: bool = False):
    item = {"item_id": item_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update(
            {"description": "This is an amazing item that has a long description"}
        )
    return item


@app.get("/itemss")
async def read_item_annotated(
    q: Annotated[
        str | None,
        Query(
            title="test",
            description="test_description",
            max_length=50,
            alias="item-query",
            deprecated=True,
        ),
    ] = None
):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})  # type: ignore
    return results


@app.get("users/{user_id}/items/{item_id}")
async def read_user_item(
    user_id: int, item_id: str, q: str | None = None, short: bool = False
):
    item = {"item_id": item_id, "owner_id": user_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update(
            {"description": "This is an amazing item that has a long description"}
        )
    return item


@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]


# Enumerations (means a fixed set of possible values)
@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet":
        return {"modelName": model_name, "message": "LeCNN all the iamges"}

    return {"model_name": model_name, "message": "Have some residuals"}


# File paths
@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    return {"file_path": file_path}
