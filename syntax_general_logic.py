from enum import Enum
from typing import Annotated, Any, List, Set

from fastapi import (
    Body,
    Cookie,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Path,
    Query,
    Response,
    UploadFile,
    status,
)
from fastapi.encoders import jsonable_encoder
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


# Status Code
# -> The status code is declared in the path operation decorator.
# -> example: @app.get("/", status_code=200)
# -> If you don't declare a status code, FastAPI will use status code 200 by default.
# -> i should use the fastapi status as it is more readable and less error-prone
# -> -> from fastapi import status
# -> -> status.HTTP_201_CREATED
# Error handling
# HTTPException(status_code=404, detail="Item not found")


# Forms
# -> Form() can be used to declare a form field.
# -> Form() is used to declare the validation and metadata for form fields.
# -> we need to define in the function parameters the form fields


# JSON Compatible Encoder from FastAPI
# -> function = jsonable_encoder()


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


class BaseUserOut(BaseUser):
    pass


class BaseUserInDB(BaseUser):
    hashed_password: str


# ------------------- Security -------------------

# OAuth2
# OpenID Connecnt used by for example Google (not OpenID, this is something else)
# look in file "security.py" for more details

# ------------------- Security END -------------------

fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]

fake_db = {}


def fake_password_hasher(raw_password: str) -> str:
    return "supersecret" + raw_password


def fake_save_user(user_in: BaseUserIn) -> BaseUserInDB:
    hashed_password = fake_password_hasher(user_in.password)
    user_in_db = BaseUserInDB(**user_in.model_dump(), hashed_password=hashed_password)
    print("User saved! ..not really")
    return user_in_db


class Tags(Enum):
    items = "items"
    users = "users"
    other = "other"
    dependency = "dependency"


# ------------------- Dependencies -------------------

# Dependencies / Dependency Injection
# -> Dependencies are used to declare the dependencies of a path operation function.
# This allows you to reuse the same logic in multiple path operations.
# https://fastapi.tiangolo.com/tutorial/dependencies/
# used for shared logic, database connections, security, authentication, role requirements, etc.
# Sub-dependencies
# -> You can also use dependencies inside other dependencies.
# They can ce as deep as you need them  to be.
# Path operations dependencies
# -> You can also use dependencies directly in the path operation decorator, if you don't need the return value of the dependency.
# you can add a list of dependencies to the path operation decorator.
# -> -> @app.get("/", dependencies=[Depends(get_db), Depends(get_current_user)])
# Global dependencies
# Similar to the way you can add dependencies to the path operation decorators, you can add them to the FastAPI application.
# -> -> app = FastAPI(dependencies=[Depends(get_token_header)])

# Dependencies with yield
# -> You can also use dependencies with yield instead of return.
# -> -> This allows you to run code before and after the path operation function.
# -> -> The code before the yield will be run before the path operation function and the code after the yield will be run after the path operation function.
# -> -> This is useful for opening and closing connections, transactions, database sessions etc.

items = {
    "foo": {"name": "Foo", "price": 50.2},
    "bar": {"name": "Bar", "description": "The bartenders", "price": 62, "tax": 20.2},
    "baz": {"name": "Baz", "description": None, "price": 50.2, "tax": 10.5, "tags": []},
}


async def common_parameters(
    q: Annotated[str | None, Body(max_length=50, description="Hello")] = None,
    skip: int = 0,
    limit: int = 100,
):
    return {"q": q, "skip": skip, "limit": limit}


CommonsDep = Annotated[dict, Depends(common_parameters)]


@app.get("/items/", status_code=status.HTTP_200_OK)
async def readdd_items(
    commons: Annotated[dict, Depends(common_parameters)],
):
    return commons


@app.get("/users/", status_code=status.HTTP_200_OK, tags=[Tags.users])
async def readddd_users(
    commons: CommonsDep,
):
    return commons


# ------


class CommonQueryParams:
    def __init__(self, q: str | None = None, skip: int = 0, limit: int = 100):
        self.q = q
        self.skip = skip
        self.limit = limit


@app.get("/items_/", status_code=status.HTTP_200_OK, tags=[Tags.dependency])
async def get_query_params(
    commons: Annotated[CommonQueryParams, Depends(CommonQueryParams)],
):
    response = {}
    if commons.q:
        response.update({"q": commons.q})
    items = fake_items_db[commons.skip : commons.skip + commons.limit]
    response.update({"items": items})

    return response


# ------


class CommonQueryParams2(BaseModel):
    q: str | None = None
    skip: int = 0
    limit: int = 100


@app.get("/items__/", status_code=status.HTTP_200_OK, tags=[Tags.dependency])
async def get_query_params2(
    commons: Annotated[CommonQueryParams2, Depends(CommonQueryParams2)],
):
    response = {}
    if commons.q:
        response.update({"q": commons.q})
    items = fake_items_db[commons.skip : commons.skip + commons.limit]
    response.update({"items": items})

    return response


CommonsDep2 = Annotated[CommonQueryParams2, Depends(CommonQueryParams2)]


@app.get("/items___/", status_code=status.HTTP_200_OK, tags=[Tags.dependency])
async def get_query_params3(
    commons: CommonsDep2,
):
    response = {}
    if commons.q:
        response.update({"q": commons.q})
    items = fake_items_db[commons.skip : commons.skip + commons.limit]
    response.update({"items": items})

    return response


# preferred way to use dependencies
@app.get("/items____/", status_code=status.HTTP_200_OK, tags=[Tags.dependency])
async def get_query_params4(
    commons: Annotated[CommonQueryParams2, Depends()],
):
    response = {}
    if commons.q:
        response.update({"q": commons.q})
    items = fake_items_db[commons.skip : commons.skip + commons.limit]
    response.update({"items": items})

    return response


# or

CommonsDep3 = Annotated[CommonQueryParams2, Depends()]


@app.get("/items______/", status_code=status.HTTP_200_OK, tags=[Tags.dependency])
async def get_query_params5(
    commons: CommonsDep3,
):
    response = {}
    if commons.q:
        response.update({"q": commons.q})
    items = fake_items_db[commons.skip : commons.skip + commons.limit]
    response.update({"items": items})

    return response


# ------


def query_extractor(q: str | None = None):
    return q


def query_or_cookie_extractor(
    q: Annotated[str, Depends(query_extractor)],
    last_query: Annotated[str | None, Cookie()] = None,
):
    if not q:
        return last_query
    return q


@app.get("/items_______/", status_code=status.HTTP_200_OK, tags=[Tags.dependency])
async def read_query(
    query_or_default: Annotated[str, Depends(query_or_cookie_extractor)],
):
    return {"q_or_cookie": query_or_default}


# ------
async def verify_token(x_token: Annotated[str, Header()]):
    if x_token != "fake-super-secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")


async def verify_key(x_key: Annotated[str, Header()]):
    if x_key != "fake-super-secret-key":
        raise HTTPException(status_code=400, detail="X-Key header invalid")
    return x_key


@app.get(
    "/items________/",
    status_code=status.HTTP_200_OK,
    tags=[Tags.dependency],
    dependencies=[Depends(verify_token), Depends(verify_key)],
)
async def read_items_with_dependencies():
    return [{"item": "Portal Gun"}, {"item": "Plumbus"}]


# -------------- Dependencies END ------------------------


@app.patch("items/{item_id}", status_code=status.HTTP_200_OK)
async def updateee_item(item_id: str, item: Item):
    # get stored item
    stored_item_data = items[item_id]  # type: ignore
    stored_item_model = Item(**stored_item_data)  # type: ignore

    # create update item with the new data
    update_data = item.model_dump(exclude_unset=True)

    # update the stored item with the new data
    updated_item = stored_item_model.model_copy(update=update_data)
    items[item_id] = jsonable_encoder(updated_item)  # type: ignore

    return updated_item


# will convert the pydantic model to a json serializable object (probable the same using model_dump(mode="json"))
@app.put("products/{id}", status_code=status.HTTP_200_OK)
async def update_product(id: int, product: Item):
    json_compatible_item_data = jsonable_encoder(product)
    fake_db[id] = json_compatible_item_data


@app.post("/login/", status_code=status.HTTP_200_OK, tags=["login"])
async def login(
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
):
    return {"username": username}


items = {1: "hello", 2: "world"}


# Description from docstring gets used as the description of the path operation in the OpenAPI schema
@app.get(
    "/itemmmmms/{item_id}",
    status_code=status.HTTP_200_OK,
    tags=["items", "login"],
    response_description="The item information",
)
async def read_itemmmm(
    item_id: Annotated[int, Path(description="The ID of the item to get")]
) -> dict:
    """
    Create an item with all the information:

    - **name**: each item must have a name
    - **description**: a long description
    - **price**: required
    - **tax**: if the item doesn't have tax, you can omit this
    - **tags**: a set of unique tag strings for this item
    """
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item_id": item_id, "item": items[item_id]}


# If you declare the type of your path operation function parameter as bytes, FastAPI will read the file for you and you will receive the contents as bytes. This will store the content in memory
@app.post("/filess/", status_code=status.HTTP_201_CREATED, tags=[Tags.other])
async def create_file(
    file: Annotated[bytes | None, File()] = None,
):
    return {"file_size": len(file)}


# https://fastapi.tiangolo.com/tutorial/request-files/
# Using UploadFile uses a spooled file, which means that the file will be stored in a temporary file and if it's small enough, it will be stored in memory otherwise it will be stored on disk
# -> UploadFile is a special class provided by FastAPI to handle file uploads.
# you get metadata about the file
# attributes: filename, content_type, file (SpooledTemporaryFile), and more
# async methods: write(data), read(size), seek[offset), close()]
@app.post("/uploadfile/", status_code=status.HTTP_201_CREATED)
async def create_upload_file(
    file: UploadFile | None = None,
):
    if file:
        return {"filename": file.filename}


@app.post("/uploadfile_with_file", status_code=status.HTTP_201_CREATED)
async def create_upload_files_with_file(
    file: Annotated[UploadFile, File(description="The file to upload")],
):
    return {"filename": file.filename}


@app.post("/uploadfiles/", status_code=status.HTTP_201_CREATED)
async def create_upload_files(
    files: Annotated[List[UploadFile], File(description="The files to upload")],
):
    return {"filenames": [(file.filename, file.content_type) for file in files]}


@app.post("/files/", status_code=status.HTTP_201_CREATED)
async def create_files(
    file: Annotated[bytes, File()],
    fileb: Annotated[UploadFile, File()],
    token: Annotated[str, Form()],
):
    return {
        "file_size": len(file),
        "fileb_content_type": fileb.content_type,
        "fileb_size": len(await fileb.read()),
        "token": token,
    }


@app.post("/user/", response_model=BaseUserOut, status_code=status.HTTP_201_CREATED)
async def create_user_(user_in: BaseUserIn):
    user_saved = fake_save_user(user_in)
    return user_saved


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
