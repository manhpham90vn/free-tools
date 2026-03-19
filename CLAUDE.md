# Python Coding Rules

## Stack
- Python 3.10+
- Formatter: Ruff (`line-length=79`)
- Type checker: mypy (strict)
- Testing: pytest

---

## Naming
- Variables & functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private members: `_single_underscore`
- Tên phải tự mô tả — không viết tắt mơ hồ (`elapsed_seconds`, không phải `es`)

## Formatting
- Indent: 4 spaces, không tab
- Dòng tối đa 79 ký tự
- 2 dòng trống giữa top-level function/class; 1 dòng trống giữa method trong class
- Không khoảng trắng thừa trong `()`, `[]`, `{}`

## Type Hints — BẮT BUỘC
- Annotate tất cả tham số và return value — không commit code thiếu type hint
- Dùng `X | Y` thay `Union[X, Y]`; `X | None` thay `Optional[X]`
- Dùng `list[str]`, `dict[str, int]` — không dùng `List`, `Dict` từ `typing`

```python
# ✅
def find_user(user_id: int, active_only: bool = True) -> dict | None: ...

# ❌
def find_user(user_id, active_only=True): ...
```

## Docstrings
- Public function/class phải có docstring — dùng **Google style**
- Mô tả ngắn trong 1 dòng đầu; thêm `Args` / `Returns` / `Raises` nếu không hiển nhiên
- Không cần docstring cho private helper rõ nghĩa

```python
def fetch_order(order_id: int, *, include_items: bool = False) -> dict:
    """Lấy thông tin đơn hàng theo ID.

    Args:
        order_id: ID đơn hàng cần truy vấn.
        include_items: Nếu True, trả về kèm danh sách sản phẩm.

    Returns:
        Dict chứa thông tin đơn hàng.

    Raises:
        OrderNotFoundError: Nếu order_id không tồn tại.
    """
```

## Functions
- Mỗi hàm chỉ làm 1 việc (Single Responsibility)
- Tối đa ~20 dòng/hàm; tối đa 5 tham số — nếu nhiều hơn, dùng dataclass
- **Return early** — thoát sớm thay vì nest nhiều `if`
- Dùng keyword-only args (`*`) cho boolean params
- KHÔNG dùng mutable default args

```python
# ✅ — return early
def process_order(order: Order) -> None:
    if not order.items:
        return
    if order.is_paid:
        return
    _charge(order)

# ❌ — nested
def process_order(order: Order) -> None:
    if order.items:
        if not order.is_paid:
            _charge(order)

# ✅ — keyword-only + no mutable default
def add_item(item: str, *, lst: list[str] | None = None) -> list[str]:
    if lst is None:
        lst = []
    lst.append(item)
    return lst
```

## Async
- Dùng `async def` / `await` nhất quán — không mix sync blocking trong async context
- I/O (HTTP, DB, file) trong async code phải dùng thư viện async (`httpx`, `asyncpg`, `aiofiles`)
- Không dùng `time.sleep()` trong async — dùng `asyncio.sleep()`
- Nhóm task độc lập bằng `asyncio.gather()`

```python
# ✅
async def fetch_users(user_ids: list[int]) -> list[dict]:
    async with httpx.AsyncClient() as client:
        tasks = [_fetch_one(client, uid) for uid in user_ids]
        return await asyncio.gather(*tasks)

# ❌ — blocking trong async
async def fetch_users(user_ids: list[int]) -> list[dict]:
    return [requests.get(f"/users/{uid}").json() for uid in user_ids]
```

## Classes
- Dùng `@dataclass` hoặc `pydantic.BaseModel` cho data containers
- Không viết `__init__` thủ công chỉ để lưu data
- Dùng `@property` thay getter/setter kiểu Java
- Dùng `@classmethod` cho alternative constructors

```python
from dataclasses import dataclass, field

@dataclass
class Order:
    order_id: int
    items: list[str] = field(default_factory=list)
    is_paid: bool = False
```

## Error Handling
- Bắt exception CỤ THỂ — không dùng bare `except:` hoặc `except Exception:`
- KHÔNG `pass` trong except — ít nhất phải log
- Dùng `raise ... from e` để giữ traceback gốc
- Custom exception kế thừa từ exception chuẩn phù hợp

```python
# ✅
class UserNotFoundError(LookupError): pass

try:
    user = fetch_user(user_id)
except httpx.TimeoutException as e:
    logger.error("Timeout for user %s: %s", user_id, e)
    raise UserNotFoundError(f"User {user_id} unreachable") from e

# ❌
try:
    user = fetch_user(user_id)
except:
    pass
```

## Imports
- Thứ tự: stdlib → third-party → local (dùng isort)
- Mỗi import 1 dòng riêng
- KHÔNG dùng wildcard: `from module import *`
- KHÔNG import bên trong function (trừ circular import)

```python
import os
from pathlib import Path

import httpx
from pydantic import BaseModel

from myapp.models import User
```

## Logging
- Dùng `logging` — KHÔNG dùng `print()` trong production
- KHÔNG dùng f-string trong log message — dùng `%s`
- Dùng `logger.exception()` trong except để giữ traceback

```python
# ✅
logger.info("Processing order %s for user %s", order_id, user_id)

# ❌
print(f"Processing order {order_id}")
logger.error(f"Error: {e}")
```

## Pythonic Patterns — Ưu tiên dùng
```python
# List comprehension
names = [u["name"] for u in users if u["active"]]

# enumerate thay range(len(...))
for idx, item in enumerate(items):
    print(idx, item)

# dict.get() thay try/except KeyError
value = data.get("key", "default")

# Walrus operator
if match := pattern.search(text):
    print(match.group())

# any() / all()
has_admin = any(u["role"] == "admin" for u in users)

# Context manager
with open("file.txt") as f:
    content = f.read()
```

## Security — KHÔNG BAO GIỜ
- Hardcode credentials/API key — dùng `os.environ` hoặc secret manager
- Dùng `eval()` / `exec()` với input từ user
- Dùng `random` cho mục đích security — dùng `secrets`
- Dùng `pickle` với data không tin tưởng

## Testing
- Mỗi public function phải có ít nhất 1 unit test
- Tên test: `test_<function>_<scenario>`
- 1 test chỉ assert 1 hành vi
- Mock external dependencies (DB, HTTP, filesystem)
- Target ≥80% coverage trên business logic

## Prohibitions
- KHÔNG để magic number — đặt tên constant
- KHÔNG comment code đã xoá — xoá hẳn
- KHÔNG dùng `type: ignore` không có lý do rõ ràng
- KHÔNG viết code thừa/không cần thiết

---

## Checklist trước khi trả lời
Trước khi output code, tự kiểm tra:
- [ ] Tất cả hàm/tham số đã có type hint?
- [ ] Không có `print()`, bare `except:`, mutable default?
- [ ] Async code không dùng blocking I/O?
- [ ] Public functions có docstring?
- [ ] Return early thay vì nested if?
- [ ] Không có magic number, wildcard import, hardcoded credential?