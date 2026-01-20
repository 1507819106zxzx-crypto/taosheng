# 武器与装备设计（v0.2）

这份文档对应当前 `game.py` 的实现：枪械（含配件改装）、弹药、近战、衣服换装。

## 1) 枪械（Guns）

> 关键参数解释  
> - `mag_size`：弹匣容量  
> - `fire_rate`：射速（发/秒）  
> - `reload_s`：换弹时间（秒）  
> - `damage`：单发伤害  
> - `bullet_speed`：子弹速度（像素/秒）  
> - `spread_deg`：散布角（度，越小越准）  
> - `noise_radius`：噪声半径（越大越容易引怪）  

### 9mm 手枪（`pistol`）
- 定位：前期可靠的“求生起手式”，稳一点、安静一点。
- 参数：`mag_size=12`，`fire_rate=4.0`，`reload_s=1.15`，`damage=12`，`bullet_speed=260`，`spread_deg=4.0`，`noise_radius=210`
- 推荐改装：
  - `mod_optic_reddot` / `mod_optic_holo`：更稳
  - `mod_muzzle_suppressor_9mm`：更安静
  - `mod_mag_ext_pistol`：更持久
  - `mod_trigger_light`：更“哒哒哒”（但更飘）

### UZI 冲锋枪（`uzi` / 9mm）
- 定位：近距离“泼水快乐枪”，火力密度高，散布也更大。
- 参数：`mag_size=32`，`fire_rate=10.5`，`reload_s=1.75`，`damage=8`，`bullet_speed=270`，`spread_deg=7.0`，`noise_radius=240`
- 推荐改装：
  - `mod_undergrip_stab`：明显更稳
  - `mod_mag_ext_9mm` 或 `mod_mag_drum_9mm`：一梭子更长（换弹更慢）
  - `mod_muzzle_suppressor_9mm`：减少“开麦打僵尸”的社交属性
  - `mod_stock_tactical`：让肩膀参与战斗

### AK-47（`ak47` / 7.62mm）
- 定位：中近距离的经典猛男枪，火力和噪声都很“讲义气”。
- 参数：`mag_size=30`，`fire_rate=8.5`，`reload_s=2.25`，`damage=18`，`bullet_speed=310`，`spread_deg=6.0`，`noise_radius=300`
- 推荐改装：
  - `mod_undergrip_stab` / `mod_undergrip_bipod`：更稳
  - `mod_muzzle_comp_rifle`：更稳但更吵
  - `mod_muzzle_suppressor_rifle` / `mod_muzzle_flash_hider_rifle`：更安静/更低调
  - `mod_mag_ext_rifle` / `mod_mag_drum_rifle`：更持久（换弹更慢）
  - `mod_optic_4x`：中距离更舒服

### SCAR-L（`scar_l` / 5.56mm）
- 定位：稳、准、帅的步枪，综合手感更“顺”。
- 参数：`mag_size=30`，`fire_rate=9.0`，`reload_s=2.10`，`damage=16`，`bullet_speed=330`，`spread_deg=5.0`，`noise_radius=290`
- 推荐改装：与 AK-47 类似，但更适配 `mod_optic_4x` 做中距离点射/短连发。

### 火箭筒（`rpg` / 火箭弹）
- 定位：清场神器，直接伤害 + 范围爆炸，噪声也会帮你“召唤全图亲戚”。
- 参数：`mag_size=1`，`fire_rate=0.65`，`reload_s=2.85`，`damage=80`，`bullet_speed=190`，`spread_deg=2.2`，`noise_radius=460`
- 特性：`bullet_kind="rocket"`，命中/撞墙/飞行超时都会爆炸：`aoe_radius=72`，`aoe_damage=55`
- 建议：留给“该跑就跑，该炸就炸”的时刻（枪店里也很稀有）。

## 2) 弹药（Ammo）
- `ammo_9mm`：手枪 / UZI
- `ammo_556`：SCAR-L
- `ammo_762`：AK-47
- `ammo_rocket`：RPG（火箭弹）

## 3) 配件与改装（Gun Mods）

### 装配规则（已实现）
- 先在背包里装备一把枪（`kind="gun"`）。
- 选中配件（`kind="gun_mod"`）并“装备”，会自动：
  - 检查兼容枪型
  - 按 `slot` 安装（同槽位旧配件会退回背包）
  - 影响枪械最终参数（弹匣容量/换弹速度/射速/散布/噪声等）

### 配件清单（按槽位）

**`optic`（瞄具）**
- `mod_optic_reddot`：通用红点（更稳）
- `mod_optic_holo`：全息（更稳）
- `mod_optic_4x`：4倍镜（步枪专用，更稳）

**`muzzle`（枪口）**
- `mod_muzzle_suppressor_9mm`：9mm 消音器（更安静）
- `mod_muzzle_suppressor_rifle`：步枪消音器（更安静）
- `mod_muzzle_comp_rifle`：补偿器（更稳但更吵）
- `mod_muzzle_flash_hider_rifle`：消焰器（更低调，偏折中）

**`under`（下挂/握把）**
- `mod_undergrip_stab`：稳定器前握把（更稳）
- `mod_undergrip_bipod`：两脚架（更稳但换弹略慢）

**`mag`（弹匣）**
- `mod_mag_ext_pistol`：手枪加长弹匣（容量 +，换弹略慢）
- `mod_mag_ext_9mm`：9mm 加长弹匣（UZI）
- `mod_mag_ext_rifle`：步枪加长弹匣（AK/SCAR）
- `mod_mag_drum_9mm`：9mm 鼓包弹匣（更大容量，换弹更慢）
- `mod_mag_drum_rifle`：步枪鼓包弹匣（更大容量，换弹更慢）

**`stock`（枪托）**
- `mod_stock_tactical`：战术枪托（更稳）

**`trigger`（扳机）**
- `mod_trigger_light`：轻量扳机组（射速更快但更飘）

## 4) 近战武器（Melee）

> `J` 为近战/拳击键：仅在“未持枪”时生效（手上拿着枪时不触发近战）。
> 想用木棍/铁管：在背包里“装备近战”，会自动收起枪；再按 `J` 挥击。

- `melee_club` 木棍：均衡、好上手
- `melee_bat` 棒球棍：范围更大、击退更强
- `melee_pipe` 铁管：更重、更痛、更吵
- `melee_machete` 砍刀：伤害更高、偏利落

## 5) 衣服与换装（Clothes / Outfit）

> 衣服是“一套造型”（上衣+裤子），当前以换装为主，不拆分部件。

换装方式（已实现）：
- 背包里选中 `kind="clothes"` 的衣服并“装备”
- 会把你当前穿的衣服退回背包（如果是开局套装，也会以衣服物品形式回收）

衣服列表（物品名 + 俏皮描述见背包底部说明）：
- `clothes_jacket_blue` / `clothes_work_green` / `clothes_tactical_gray` / `clothes_prisoner_orange` / `clothes_medic_white`
- `clothes_raincoat_yellow` / `clothes_hoodie_pink` / `clothes_denim_blue` / `clothes_chef_white`
- `clothes_racing_red` / `clothes_xmas_green` / `clothes_black_suit` / `clothes_desert_camo`
- `clothes_sport_cyan` / `clothes_pajama_bear` / `clothes_maid`

## 6) 操作速查（Hardcore 生存原型）
- 移动：`WASD` / 方向键
- 冲刺：`Shift`（已修复斜向更快问题，并调慢冲刺速度）
- 拾取：`E`
- 射击：鼠标左键（需要先装备枪）
- 换弹：`R`
- 近战：`J`（未持枪时）
- 背包：`Tab`（选中物品后可装枪/装配件/换衣服/换近战；`Q` 丢弃）
- 载具：`F` 上/下车（优先最近的房车/自行车）
